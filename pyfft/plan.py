import numpy

from .kernel import GlobalFFTKernel, LocalFFTKernel, X_DIRECTION, Y_DIRECTION, Z_DIRECTION
from .kernel_helpers import log2, getRadixArray

_FFT_1D = 1
_FFT_2D = 2
_FFT_3D = 3

class _FFTParams:
	"""
	Internal class, which serves as an interface between kernel creator and plan.
	Contains different device and FFT plan parameters.
	"""

	def __init__(self, shape, dtype, context):

		self.x, self.y, self.z = shape
		self.size = self.x * self.y * self.z
		self.context = context

		if 2 ** log2(self.size) != self.size:
			raise ValueError("Array dimensions must be powers of two")

		if dtype in [numpy.complex64, numpy.float32]:
			self.split = True if dtype == numpy.float32 else False
			self.scalar = 'float'
			self.complex = 'float2'
			self.scalar_nbytes = 4
			self.complex_nbytes = 8
		elif dtype in [numpy.complex128, numpy.float64]:
			self.split = True if dtype == numpy.float64 else False
			self.scalar = 'double'
			self.complex = 'double2'
			self.scalar_nbytes = 8
			self.complex_nbytes = 16
		else:
			raise ValueError("Data type " + str(dtype) + " is not supported")

		global_memory_word = self.scalar_nbytes if self.split else self.complex_nbytes

		self.min_mem_coalesce_width = context.min_mem_coalesce_width[global_memory_word]
		self.num_smem_banks = context.num_smem_banks
		self.max_grid_x = context.max_grid_x
		self.max_block_size = context.max_block_size

		self.max_smem_fft_size = 2048
		self.max_radix = 16


class FFTPlan:
	"""
	Class for FFT plan preparation and execution.
	"""
	def __init__(self, context, shape, dtype=numpy.complex64, normalize=True, wait_for_finish=None):

		if isinstance(shape, int):
			self._dim = _FFT_1D
			shape = (shape, 1, 1)
		elif isinstance(shape, tuple):
			if len(shape) == 1:
				self._dim = _FFT_1D
				shape = (shape[0], 1, 1)
			elif len(shape) == 2:
				self._dim = _FFT_2D
				shape = (shape[1], shape[0], 1)
			elif len(shape) == 3:
				self._dim = _FFT_3D
				shape = tuple(reversed(shape))
			else:
				raise ValueError("Wrong shape")
		else:
			raise ValueError("Wrong shape")

		self._context = context
		self._params = _FFTParams(shape, dtype, context)
		self._normalize = normalize
		self._wait_for_finish = wait_for_finish

		self._tempmemobj = None
		self._tempmemobj_re = None
		self._tempmemobj_im = None

		# prepared functions and temporary buffers are cached for repeating batch sizes
		self._last_batch_size = 0

		if self._params.split:
			self.execute = self._executeSplit
		else:
			self.execute = self._executeInterleaved

		self._generateKernelCode()

	def _generateKernelCode(self):
		"""Create and compile FFT kernels"""

		self._kernels = []
		if self._dim == _FFT_1D:
			self._kernels.extend(self._fft1D(X_DIRECTION))
		elif self._dim == _FFT_2D:
			self._kernels.extend(self._fft1D(X_DIRECTION))
			self._kernels.extend(self._fft1D(Y_DIRECTION))
		else:
			self._kernels.extend(self._fft1D(X_DIRECTION))
			self._kernels.extend(self._fft1D(Y_DIRECTION))
			self._kernels.extend(self._fft1D(Z_DIRECTION))

		# Since we're changing the last kernel, it won't affect
		# 'chaining' of batch sizes in global kernels
		if self._normalize:
			self._kernels[-1].addNormalization()

		self._temp_buffer_needed = False
		for kernel in self._kernels:
			if not kernel.in_place_possible:
				self._temp_buffer_needed = True

	def _fft1D(self, dir):
		"""Create and compile kernels for one of the dimensions"""

		kernels = []

		if dir == X_DIRECTION:
			if self._params.x > self._params.max_smem_fft_size:
				kernels.extend(GlobalFFTKernel.createChain(self._params,
					self._params.x, 1, X_DIRECTION, 1))
			elif self._params.x > 1:
				radix_array = getRadixArray(self._params.x, 0)
				if self._params.x / radix_array[0] <= self._params.max_block_size:
					kernel = LocalFFTKernel(self._params, self._params.x)
					kernel.compile(self._params.max_block_size)
					kernels.append(kernel)
				else:
					radix_array = getRadixArray(self._params.x, self._params.max_radix)
					if self._params.x / radix_array[0] <= self._params.max_block_size:
						kernel = LocalFFTKernel(self._params, self._params.x)
						kernel.compile(self._params.max_block_size)
						kernels.append(kernel)
					else:
						# TODO: find out which conditions are necessary to execute this code
						kernels.extend(GlobalFFTKernel.createChain(self._params,
							self._params.x, 1 , X_DIRECTION, 1))
		elif dir == Y_DIRECTION:
			if self._params.y > 1:
				kernels.extend(GlobalFFTKernel.createChain(self._params, self._params.y,
					self._params.x, Y_DIRECTION, 1))
		elif dir == Z_DIRECTION:
			if self._params.z > 1:
				kernels.extend(GlobalFFTKernel.createChain(self._params, self._params.z,
					self._params.x * self._params.y, Z_DIRECTION, 1))
		else:
			raise ValueError("Wrong direction")

		return kernels

	def _execute(self, wait_for_finish, is_inplace, inverse, batch, *args):
		"""Execute plan for given data type"""

		inplace_done = False
		split = self._params.split

		new_batch = False
		if self._last_batch_size != batch:
			self._last_batch_size = batch
			new_batch = True

		if self._temp_buffer_needed and new_batch:
			self._last_batch_size = batch
			buffer_size = self._params.size * batch * self._params.scalar_nbytes

			if split:
				self._tempmemobj_re = self._context.allocate(buffer_size)
				self._tempmemobj_im = self._context.allocate(buffer_size)
			else:
				self._tempmemobj = self._context.allocate(buffer_size * 2)

		if split:
			mem_objs_re = (args[0], args[2], self._tempmemobj_re)
			mem_objs_im = (args[1], args[3], self._tempmemobj_im)
		else:
			mem_objs = (args[0], args[1], self._tempmemobj)

		num_kernels_is_odd = (len(self._kernels) % 2 == 1)
		curr_read  = 0
		curr_write = 1

		self._context.createQueue()

		# at least one external dram shuffle (transpose) required
		inplace_done = False
		if self._temp_buffer_needed:
			# in-place transform
			if is_inplace:
				curr_read  = 1
				curr_write = 2
				inplace_done = False
			else:
				curr_write = 1 if num_kernels_is_odd else 2

			for kernel in self._kernels:
				if is_inplace and num_kernels_is_odd and not inplace_done and kernel.in_place_possible:
					curr_write = curr_read
					inplace_done = True

				if new_batch:
					kernel.prepare(batch)

				if split:
					self._context.enqueue(kernel, inverse, mem_objs_re[curr_read], mem_objs_im[curr_read],
						mem_objs_re[curr_write], mem_objs_im[curr_write])
				else:
					self._context.enqueue(kernel, inverse, mem_objs[curr_read], mem_objs[curr_write])

				curr_read  = 1 if (curr_write == 1) else 2
				curr_write = 2 if (curr_write == 1) else 1

		# no dram shuffle (transpose required) transform
		# all kernels can execute in-place.
		else:
			for kernel in self._kernels:
				if new_batch:
					kernel.prepare(batch)

				if split:
					self._context.enqueue(kernel, inverse, mem_objs_re[curr_read], mem_objs_im[curr_read],
						mem_objs_re[curr_write], mem_objs_im[curr_write])
				else:
					self._context.enqueue(kernel, inverse, mem_objs[curr_read], mem_objs[curr_write])

				curr_read  = 1
				curr_write = 1

		# global wait setting has lower priority than the local one
		wait = self._wait_for_finish
		if wait_for_finish is not None:
			wait = wait_for_finish

		if wait:
			self._context.wait()
		else:
			self._context.flush()
			return self._context.getQueue()

	def _executeInterleaved(self, data_in, data_out=None, inverse=False, batch=1, wait_for_finish=None):
		"""Execute plan for interleaved complex array"""

		if data_out is None:
			data_out = data_in
			is_inplace = True
		else:
			is_inplace = False

		return self._execute(wait_for_finish, is_inplace, inverse,
			batch, data_in, data_out)

	def _executeSplit(self, data_in_re, data_in_im, data_out_re=None, data_out_im=None, inverse=False, batch=1, wait_for_finish=None):
		"""Execute plan for split complex array"""

		if data_out_re is None and data_out_im is None:
			data_out_re = data_in_re
			data_out_im = data_in_im
			is_inplace = True
		else:
			is_inplace = False

		return self._execute(wait_for_finish, is_inplace, inverse,
			batch, data_in_re, data_in_im, data_out_re, data_out_im)
