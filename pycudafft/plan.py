from pycuda.autoinit import device
from pycuda.compiler import SourceModule
from pycuda.driver import device_attribute
from pycuda.gpuarray import GPUArray
import pycuda.driver as cuda

from kernel import *

_FFT_1D = 1
_FFT_2D = 2
_FFT_3D = 3

class FFTPlan:

	def __init__(self, x, y=None, z=None, split=False):

		# TODO: check that dimensions are the power of two
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

		self.split = split
		self.n = _Dim(x, y, z)
		self.temp_buffer_needed = False
		self.last_batch_size = 0
		self.tempmemobj = None
		self.tempmemobj_re = None
		self.tempmemobj_im = None
		self.max_smem_fft_size = 2048
		self.max_radix = 16
		self.min_mem_coalesce_width = 16

		# TODO: get this parameter properly from device instead of calculating it
		self.num_smem_banks = device.get_attribute(device_attribute.WARP_SIZE) / 2

		self.max_block_size = device.get_attribute(device_attribute.MAX_BLOCK_DIM_X)

		# TODO: make this 'recompile-if-necessary' code more good looking
		done = False
		while not done:
			self.kernels = []
			self._generateKernelCode()
			try:
				self._compileKernels()
			except:
				if self.max_block_size > 1:
					self.max_block_size /= 2
					continue
				raise Exception("Cannot meet number of registers/shared memory requirements")
			done = True

	def _compileKernels(self):
		shared_mem_limit = device.get_attribute(device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
		reg_limit = device.get_attribute(device_attribute.MAX_REGISTERS_PER_BLOCK)

		for kinfo in self.kernels:
			kinfo.module = SourceModule(kinfo.kernel_string, no_extern_c=True)
			kinfo.function_ref = kinfo.module.get_function(kinfo.kernel_name)
			if kinfo.function_ref.shared_size_bytes > shared_mem_limit:
				raise Exception("Insufficient shared memory")
			if kinfo.function_ref.num_regs * kinfo.block_size > reg_limit:
				raise Exception("Insufficient registers")

	def _generateKernelCode(self):

		if self.dim == _FFT_1D:
			self.kernels.extend(FFT1D(self, X_DIRECTION))
		elif self.dim == _FFT_2D:
			self.kernels.extend(FFT1D(self, X_DIRECTION))
			self.kernels.extend(FFT1D(self, Y_DIRECTION))
		else:
			self.kernels.extend(FFT1D(self, X_DIRECTION))
			self.kernels.extend(FFT1D(self, Y_DIRECTION))
			self.kernels.extend(FFT1D(self, Z_DIRECTION))

		self.temp_buffer_needed = False
		for kinfo in self.kernels:
			if not kinfo.in_place_possible:
				self.temp_buffer_needed = True

	def _getKernelWorkDimensions(self, kinfo, batch):

		block_size = kinfo.block_size
		blocks_num = kinfo.blocks_num

		if kinfo.dir == X_DIRECTION:
			max_smem_fft_size = self.max_smem_fft_size
			if self.n.x <= max_smem_fft_size:
				batch = self.n.y * self.n.z * batch
				blocks_num = (batch / blocks_num + 1) if batch % blocks_num != 0 else batch / blocks_num
			else:
				batch *= self.n.y * self.n.z
				blocks_num *= batch
		elif kinfo.dir == Y_DIRECTION:
			batch *= self.n.z
			blocks_num *= batch
		elif kinfo.dir == Z_DIRECTION:
			blocks_num *= batch

		return batch, blocks_num, block_size

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

		dir = 1 if inverse else -1

		if self.temp_buffer_needed and self.last_batch_size != batch:
			self.last_batch_size = batch
			# TODO: remove hardcoded '2 * 4' when adding support for different types
			self.tempmemobj = cuda.mem_alloc(self.n.x * self.n.y * self.n.z * batch * 2 * 4)

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

				s = batch
				s, blocks_num, block_size = self._getKernelWorkDimensions(kinfo, s)

				func = kinfo.function_ref
				# TODO: prepare functions when creating the plan
				func.prepare("PPii", block=(block_size, 1, 1))
				func.prepared_call((blocks_num, 1), mem_objs[curr_read], mem_objs[curr_write], dir, s)

				curr_read  = 1 if (curr_write == 1) else 2
				curr_write = 2 if (curr_write == 1) else 1

		# no dram shuffle (transpose required) transform
		# all kernels can execute in-place.
		else:
			for kinfo in self.kernels:

				s = batch
				s, blocks_num, block_size = self._getKernelWorkDimensions(kinfo, s)

				func = kinfo.function_ref
				# TODO: prepare functions when creating the plan
				func.prepare("PPii", block=(block_size, 1, 1))
				func.prepared_call((blocks_num, 1), mem_objs[curr_read], mem_objs[curr_write], dir, s)

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

		dir = 1 if inverse else -1

		if self.temp_buffer_needed and self.last_batch_size != batch:
			self.last_batch_size = batch
			# TODO: remove hardcoded '4' when adding support for different types
			self.tempmemobj_re = cuda.mem_alloc(self.n.x * self.n.y * self.n.z * batch * 4)
			self.tempmemobj_im = cuda.mem_alloc(self.n.x * self.n.y * self.n.z * batch * 4)

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

				s = batch
				s, blocks_num, block_size = self._getKernelWorkDimensions(kinfo, s)

				func = kinfo.function_ref
				# TODO: prepare functions when creating the plan
				func.prepare("PPPPii", block=(block_size, 1, 1))
				func.prepared_call((blocks_num, 1), mem_objs_re[curr_read],
					mem_objs_im[curr_read], mem_objs_re[curr_write], mem_objs_im[curr_write], dir, s)

				curr_read  = 1 if (curr_write == 1) else 2
				curr_write = 2 if (curr_write == 1) else 1

		# no dram shuffle (transpose required) transform
		# all kernels can execute in-place.
		else:
			for kinfo in self.kernels:

				s = batch
				s, blocks_num, block_size = self._getKernelWorkDimensions(kinfo, s)

				func = kinfo.function_ref
				# TODO: prepare functions when creating the plan
				func.prepare("PPPPii", block=(block_size, 1, 1))
				func.prepared_call((blocks_num, 1), mem_objs_re[curr_read],
					mem_objs_im[curr_read], mem_objs_re[curr_write], mem_objs_im[curr_write], dir, s)

				curr_read  = 1
				curr_write = 1
