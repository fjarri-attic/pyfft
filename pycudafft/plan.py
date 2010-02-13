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
		# and number of elements in n corresponds to dim

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
		self.num_kernels = 0
		self.program = 0
		self.temp_buffer_needed = False
		self.last_batch_size = 0
		self.tempmemobj = None
		self.tempmemobj_re = None
		self.tempmemobj_im = None
		self.max_localmem_fft_size = 2048
		self.max_radix = 16
		self.min_mem_coalesce_width = 16

		# TODO: get this parameter properly from device instead of calculating it
		self.num_local_mem_banks = device.get_attribute(device_attribute.WARP_SIZE) / 2

		self.max_work_item_per_workgroup = device.get_attribute(device_attribute.MAX_BLOCK_DIM_X)

		# TODO: make this 'recompile-if-necessary' code more good looking
		done = False
		while not done:
			self.kernels = []
			self.getBlockConfigAndKernelString()
			try:
				self.compileKernels()
			except:
				if self.max_work_item_per_workgroup > 1:
					self.max_work_item_per_workgroup /= 2
					continue
				raise Exception("Cannot meet number of registers/shared memory requirements")
			done = True

	def compileKernels(self):
		shared_mem_limit = device.get_attribute(device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
		reg_limit = device.get_attribute(device_attribute.MAX_REGISTERS_PER_BLOCK)

		for kInfo in self.kernels:
			kInfo.module = SourceModule(kInfo.kernel_string, no_extern_c=True)
			kInfo.function_ref = kInfo.module.get_function(kInfo.kernel_name)
			if kInfo.function_ref.shared_size_bytes > shared_mem_limit:
				raise Exception("Insufficient shared memory")
			if kInfo.function_ref.num_regs * kInfo.num_workitems_per_workgroup > reg_limit:
				raise Exception("Insufficient registers")

	def getBlockConfigAndKernelString(self):

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
		for kInfo in self.kernels:
			if not kInfo.in_place_possible:
				self.temp_buffer_needed = True

	def getKernelWorkDimensions(self, kernelInfo, batch):

		lWorkItems = kernelInfo.num_workitems_per_workgroup
		numWorkGroups = kernelInfo.num_workgroups

		if kernelInfo.dir == X_DIRECTION:
			maxLocalMemFFTSize = self.max_localmem_fft_size
			if self.n.x <= maxLocalMemFFTSize:
				batch = self.n.y * self.n.z * batch
				numWorkGroups = (batch / numWorkGroups + 1) if batch % numWorkGroups != 0 else batch / numWorkGroups
			else:
				batch *= self.n.y * self.n.z
				numWorkGroups *= batch
		elif kernelInfo.dir == Y_DIRECTION:
			batch *= self.n.z
			numWorkGroups *= batch
		elif kernelInfo.dir == Z_DIRECTION:
			numWorkGroups *= batch

		gWorkItems = numWorkGroups * lWorkItems
		return batch, gWorkItems, lWorkItems

	def execute(self, data_in, data_out=None, inverse=False, batch=1):

		inPlaceDone = 0
		if data_out is None:
			data_out = data_in
			isInPlace = True
		else:
			isInPlace = False

		if isinstance(data_in, GPUArray):
			data_in = data_in.gpudata

		if isinstance(data_out, GPUArray):
			data_out = data_out.gpudata

		dir = 1 if inverse else -1

		if self.temp_buffer_needed and self.last_batch_size != batch:
			self.last_batch_size = batch
			# TODO: remove hardcoded '2 * 4' when adding support for different types
			self.tempmemobj = cuda.mem_alloc(self.n.x * self.n.y * self.n.z * batch * 2 * 4)

		memObj = (data_in, data_out, self.tempmemobj)

		numKernels = len(self.kernels)

		numKernelsOdd = (numKernels % 2 == 1)
		currRead  = 0
		currWrite = 1

		# at least one external dram shuffle (transpose) required
		inPlaceDone = False
		if self.temp_buffer_needed:
			# in-place transform
			if isInPlace:
				currRead  = 1
				currWrite = 2
			else:
				currWrite = 1 if numKernelsOdd else 2

			for kInfo in self.kernels:
				if isInPlace and numKernelsOdd and not inPlaceDone and kInfo.in_place_possible:
					currWrite = currRead
					inPlaceDone = True

				s = batch
				s, gWorkItems, lWorkItems = self.getKernelWorkDimensions(kInfo, s)

				func = kInfo.function_ref
				# TODO: prepare functions when creating the plan
				func.prepare("PPii", block=(lWorkItems, 1, 1))
				func.prepared_call((gWorkItems / lWorkItems, 1), memObj[currRead], memObj[currWrite], dir, s)

				currRead  = 1 if (currWrite == 1) else 2
				currWrite = 2 if (currWrite == 1) else 1

		# no dram shuffle (transpose required) transform
		# all kernels can execute in-place.
		else:
			for kInfo in self.kernels:

				s = batch
				s, gWorkItems, lWorkItems = self.getKernelWorkDimensions(kInfo, s)

				func = kInfo.function_ref
				# TODO: prepare functions when creating the plan
				func.prepare("PPii", block=(lWorkItems, 1, 1))
				func.prepared_call((gWorkItems / lWorkItems, 1), memObj[currRead], memObj[currWrite], dir, s)

				currRead  = 1
				currWrite = 1

	def executeSplit(self, data_in_re, data_in_im, data_out_re=None, data_out_im=None, inverse=False, batch=1):

		inPlaceDone = 0
		if data_out_re is None and data_out_im is None:
			data_out_re = data_in_re
			data_out_im = data_in_im
			isInPlace = True
		else:
			isInPlace = False

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

		memObj_re = (data_in_re, data_out_re, self.tempmemobj_re)
		memObj_im = (data_in_im, data_out_im, self.tempmemobj_im)

		numKernels = len(self.kernels)

		numKernelsOdd = (numKernels % 2 == 1)
		currRead  = 0
		currWrite = 1

		# at least one external dram shuffle (transpose) required
		inPlaceDone = False
		if self.temp_buffer_needed:
			# in-place transform
			if isInPlace:
				currRead  = 1
				currWrite = 2
			else:
				currWrite = 1 if numKernelsOdd else 2

			for kInfo in self.kernels:
				if isInPlace and numKernelsOdd and not inPlaceDone and kInfo.in_place_possible:
					currWrite = currRead
					inPlaceDone = True

				s = batch
				s, gWorkItems, lWorkItems = self.getKernelWorkDimensions(kInfo, s)

				func = kInfo.function_ref
				# TODO: prepare functions when creating the plan
				func.prepare("PPPPii", block=(lWorkItems, 1, 1))
				func.prepared_call((gWorkItems / lWorkItems, 1), memObj_re[currRead],
					memObj_im[currRead], memObj_re[currWrite], memObj_im[currWrite], dir, s)

				currRead  = 1 if (currWrite == 1) else 2
				currWrite = 2 if (currWrite == 1) else 1

		# no dram shuffle (transpose required) transform
		# all kernels can execute in-place.
		else:
			for kInfo in self.kernels:

				s = batch
				s, gWorkItems, lWorkItems = self.getKernelWorkDimensions(kInfo, s)

				func = kInfo.function_ref
				# TODO: prepare functions when creating the plan
				func.prepare("PPPPii", block=(lWorkItems, 1, 1))
				func.prepared_call((gWorkItems / lWorkItems, 1), memObj_re[currRead],
					memObj_im[currRead], memObj_re[currWrite], memObj_im[currWrite], dir, s)

				currRead  = 1
				currWrite = 1
