from pycuda.autoinit import device
from pycuda.compiler import SourceModule
from pycuda.driver import device_attribute
from pycuda.gpuarray import GPUArray
import pycuda.driver as cuda
from pycuda.tools import DeviceData

import numpy

from .kernel import *
from .kernel_helpers import log2

_FFT_1D = 1
_FFT_2D = 2
_FFT_3D = 3

SINGLE_PRECISION = 0

class FFTPlan:

	def __init__(self, x, y=None, z=None, split=False, precision=SINGLE_PRECISION, mempool=None):

		if z is None:
			if y is None:
				self.dim = _FFT_1D
			else:
				self.dim = _FFT_2D
		else:
			self.dim = _FFT_3D

		class _Dim:
			def __init__(self, x, y, z):
				self.x = x
				self.y = 1 if y is None else y
				self.z = 1 if z is None else z

		if precision == SINGLE_PRECISION:
			self.scalar = 'float'
			self.complex = 'float2'
			self.scalar_nbytes = 4
			self.complex_nbytes = 8
		else:
			raise ValueError("Precision is not supported: " + str(precision))

		if mempool is None:
			self.allocate = cuda.mem_alloc
		else:
			self.allocate = mempool.allocate

		self.tempmemobj = None
		self.tempmemobj_re = None
		self.tempmemobj_im = None

		self.split = split
		self.n = _Dim(x, y, z)
		for n in (self.n.x, self.n.y, self.n.z):
			if 2 ** log2(n) != n:
				raise ValueError("Array dimensions must be powers of two")

		self.last_batch_size = 0
		self.max_smem_fft_size = 2048
		self.max_radix = 16

		self.devdata = DeviceData()

		global_memory_word = self.scalar_nbytes if split else self.complex_nbytes
		self.min_mem_coalesce_width = self.devdata.align_bytes(word_size=global_memory_word) / global_memory_word

		# TODO: get this parameter properly from device instead of calculating it
		self.num_smem_banks = self.devdata.smem_granularity

		self.max_block_size = device.get_attribute(device_attribute.MAX_BLOCK_DIM_X)
		self.max_registers = device.get_attribute(device_attribute.MAX_REGISTERS_PER_BLOCK)

		self._generateKernelCode()

	def _generateKernelCode(self):
		self.kernels = []
		if self.dim == _FFT_1D:
			self.kernels.extend(self._fft1D(X_DIRECTION))
		elif self.dim == _FFT_2D:
			self.kernels.extend(self._fft1D(X_DIRECTION))
			self.kernels.extend(self._fft1D(Y_DIRECTION))
		else:
			self.kernels.extend(self._fft1D(X_DIRECTION))
			self.kernels.extend(self._fft1D(Y_DIRECTION))
			self.kernels.extend(self._fft1D(Z_DIRECTION))

		self.temp_buffer_needed = False
		for kinfo in self.kernels:
			if not kinfo.in_place_possible:
				self.temp_buffer_needed = True

	def _fft1D(self, dir):

		kernels = []

		if dir == X_DIRECTION:
			if self.n.x > self.max_smem_fft_size:
				kernels.extend(GlobalFFTKernel.createChain(self, self.n.x, 1 , X_DIRECTION, 1))
			elif self.n.x > 1:
				radix_array = getRadixArray(self.n.x, 0)
				if self.n.x / radix_array[0] <= self.max_block_size:
					kernel = LocalFFTKernel(self, self.n.x)
					kernel.compile(self.max_block_size)
					kernels.append(kernel)
				else:
					radix_array = getRadixArray(self.n.x, self.max_radix)
					if self.n.x / radix_array[0] <= self.max_block_size:
						kernel = LocalFFTKernel(self, self.n.x)
						kernel.compile(self.max_block_size)
						kernels.append(kernel)
					else:
						# TODO: find out which conditions are necessary to execute this code
						kernels.extend(GlobalFFTKernel.createChain(self, self.n.x, 1 , X_DIRECTION, 1))
		elif dir == Y_DIRECTION:
			if self.n.y > 1:
				kernels.extend(GlobalFFTKernel.createChain(self, self.n.y, self.n.x, Y_DIRECTION, 1))
		elif dir == Z_DIRECTION:
			if self.n.z > 1:
				kernels.extend(GlobalFFTKernel.createChain(self, self.n.z, self.n.x * self.n.y, Z_DIRECTION, 1))
		else:
			raise ValueError("Wrong direction")

		return kernels

	def execute(self, data_in, data_out=None, inverse=False, batch=1):

		inplace_done = False
		if data_out is None:
			data_out = data_in
			is_inplace = True
		else:
			is_inplace = False

		if isinstance(data_in, GPUArray):
			data_in = data_in.gpudata

		if isinstance(data_out, GPUArray):
			data_out = data_out.gpudata

		new_batch = False
		if self.last_batch_size != batch:
			self.last_batch_size = batch
			new_batch = True

		if self.temp_buffer_needed and new_batch:
			self.last_batch_size = batch
			self.tempmemobj = self.allocate(self.n.x * self.n.y * self.n.z * batch * self.complex_nbytes)

		mem_objs = (data_in, data_out, self.tempmemobj)

		num_kernels = len(self.kernels)

		num_kernels_is_odd = (num_kernels % 2 == 1)
		curr_read  = 0
		curr_write = 1

		# at least one external dram shuffle (transpose) required
		inplace_done = False
		if self.temp_buffer_needed:
			# in-place transform
			if is_inplace:
				curr_read  = 1
				curr_write = 2
				inplace_done = False
			else:
				curr_write = 1 if num_kernels_is_odd else 2

			for kinfo in self.kernels:
				if is_inplace and num_kernels_is_odd and not inplace_done and kinfo.in_place_possible:
					curr_write = curr_read
					inplace_done = True

				if new_batch:
					kinfo.prepare(batch)
				kinfo.preparedCall(mem_objs[curr_read], mem_objs[curr_write], inverse)

				curr_read  = 1 if (curr_write == 1) else 2
				curr_write = 2 if (curr_write == 1) else 1

		# no dram shuffle (transpose required) transform
		# all kernels can execute in-place.
		else:
			for kinfo in self.kernels:
				if new_batch:
					kinfo.prepare(batch)
				kinfo.preparedCall(mem_objs[curr_read], mem_objs[curr_write], inverse)

				curr_read  = 1
				curr_write = 1

	def executeSplit(self, data_in_re, data_in_im, data_out_re=None, data_out_im=None, inverse=False, batch=1):

		inplace_done = False
		if data_out_re is None and data_out_im is None:
			data_out_re = data_in_re
			data_out_im = data_in_im
			is_inplace = True
		else:
			is_inplace = False

		if isinstance(data_in_re, GPUArray):
			data_in_re = data_in_re.gpudata

		if isinstance(data_in_im, GPUArray):
			data_in_im = data_in_im.gpudata

		if isinstance(data_out_re, GPUArray):
			data_out_re = data_out_re.gpudata

		if isinstance(data_out_im, GPUArray):
			data_out_im = data_out_im.gpudata

		new_batch = False
		if self.last_batch_size != batch:
			self.last_batch_size = batch
			new_batch = True

		if self.temp_buffer_needed and new_batch:
			self.last_batch_size = batch
			self.tempmemobj_re = self.allocate(self.n.x * self.n.y * self.n.z * batch * self.scalar_nbytes)
			self.tempmemobj_im = self.allocate(self.n.x * self.n.y * self.n.z * batch * self.scalar_nbytes)

		mem_objs_re = (data_in_re, data_out_re, self.tempmemobj_re)
		mem_objs_im = (data_in_im, data_out_im, self.tempmemobj_im)

		num_kernels = len(self.kernels)

		num_kernels_is_odd = (num_kernels % 2 == 1)
		curr_read  = 0
		curr_write = 1

		# at least one external dram shuffle (transpose) required
		inplace_done = False
		if self.temp_buffer_needed:
			# in-place transform
			if is_inplace:
				curr_read  = 1
				curr_write = 2
				inplace_done = False
			else:
				curr_write = 1 if num_kernels_is_odd else 2

			for kinfo in self.kernels:
				if is_inplace and num_kernels_is_odd and not inplace_done and kinfo.in_place_possible:
					curr_write = curr_read
					inplace_done = True

				if new_batch:
					kinfo.prepare(batch)
				kinfo.preparedCallSplit(mem_objs_re[curr_read], mem_objs_im[curr_read],
					mem_objs_re[curr_write], mem_objs_im[curr_write], inverse)

				curr_read  = 1 if (curr_write == 1) else 2
				curr_write = 2 if (curr_write == 1) else 1

		# no dram shuffle (transpose required) transform
		# all kernels can execute in-place.
		else:
			for kinfo in self.kernels:
				if new_batch:
					kinfo.prepare(batch)
				kinfo.preparedCallSplit(mem_objs_re[curr_read], mem_objs_im[curr_read],
					mem_objs_re[curr_write], mem_objs_im[curr_write], inverse)

				curr_read  = 1
				curr_write = 1
