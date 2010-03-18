import os.path

from mako.template import Template

from .kernel_helpers import *

X_DIRECTION = 0
Y_DIRECTION = 1
Z_DIRECTION = 2

_dir, _file = os.path.split(os.path.abspath(__file__))
_template = Template(filename=os.path.join(_dir, 'kernel.mako'))

class _FFTKernel:
	"""Base class for FFT kernels. Handles compilation and execution."""

	def __init__(self, fft_params):
		self._params = fft_params
		self._context = fft_params.context
		self._kernel_name = "fft"
		self._previous_batch = None

	def addNormalization(self):
		kernel_string = self.generate(self._max_block_size, normalize=True)

		# using the last successful max_block_size, because normalization clearly
		# will not decrease the number of used registers
		self.compile(self._max_block_size, normalize=True)

	def compile(self, max_block_size, normalize=False):
		self._module = None
		self._func_ref = None
		max_block_size = max_block_size * 2

		# starting from the maximum available block size, generate and compile kernels
		# stop when number of registers is less than maximum available for block
		while max_block_size > self._params.num_smem_banks:
			max_block_size /= 2

			# Try to generate kernel code. Assertion error means that
			# given parameters do not allow us to create code;
			# other errors are not expected and passed further
			try:
				kernel_string = self.generate(max_block_size, normalize)
			except AssertionError as e:
				continue

			# compile and get function pointers
			module = self._context.compile(kernel_string)
			func_forward = module.getFunction(self._kernel_name + "Fwd", self._params.split, self._block_size)
			func_inverse = module.getFunction(self._kernel_name + "Inv", self._params.split, self._block_size)

			# check that kernel is executable (testing only one, because the other
			# is exactly the same)
			if not func_forward.isExecutable():
				continue

			self._module = module
			self._func_forward = func_forward
			self._func_inverse = func_inverse
			self._max_block_size = max_block_size
			break

		if self._module is None:
			raise Exception("Failed to find block size for the kernel")

	def prepare(self, batch):
		"""Prepare function call. Caches prepared functions for repeating batch sizes."""

		if self._previous_batch != batch:
			self._previous_batch = batch
			batch, grid = self._getKernelWorkDimensions(batch)

			self._func_forward.prepare(grid, batch)
			self._func_inverse.prepare(grid, batch)

	def __call__(self, queue, inverse, *args):
		func = self._func_inverse if inverse else self._func_forward
		func(queue, *args)

	def _getKernelWorkDimensions(self, batch):
		blocks_num = self._blocks_num

		if self._dir == X_DIRECTION:
			if self._params.x <= self._params.max_smem_fft_size:
				batch = self._params.y * self._params.z * batch
				blocks_num = (batch / blocks_num + 1) if batch % blocks_num != 0 else batch / blocks_num
			else:
				batch *= self._params.y * self._params.z
				blocks_num *= batch
		elif self._dir == Y_DIRECTION:
			batch *= self._params.z
			blocks_num *= batch
		elif self._dir == Z_DIRECTION:
			blocks_num *= batch

		if blocks_num > self._params.max_grid_x:
			grid = (self._params.max_grid_x, self._params.max_grid_x / blocks_num)
		else:
			grid = (blocks_num, 1)

		return batch, grid


class LocalFFTKernel(_FFTKernel):
	"""Generator for 'local' FFT in shared memory"""

	def __init__(self, fft_params, n):
		_FFTKernel.__init__(self, fft_params)
		self._n = n

	def generate(self, max_block_size, normalize):
		n = self._n
		assert n <= max_block_size * self._params.max_radix, "Signal length is too big for shared mem fft"

		radix_array = getRadixArray(n, 0)
		if n / radix_array[0] > max_block_size:
			radix_array = getRadixArray(n, self._params.max_radix)

		assert radix_array[0] <= self._params.max_radix, "Max radix choosen is greater than allowed"
		assert n / radix_array[0] <= max_block_size, \
			"Required number of threads per xform greater than maximum block size for local mem fft"

		self._dir = X_DIRECTION
		self.in_place_possible = True

		threads_per_xform = n / radix_array[0]
		block_size = 64 if threads_per_xform <= 64 else threads_per_xform
		assert block_size <= max_block_size
		xforms_per_block = block_size / threads_per_xform
		self._blocks_num = xforms_per_block
		self._block_size = block_size

		smem_size = getSharedMemorySize(n, radix_array, threads_per_xform, xforms_per_block,
			self._params.num_smem_banks, self._params.min_mem_coalesce_width)

		return _template.get_def("localKernel").render(
			self._params.scalar, self._params.complex, self._params.split, self._kernel_name,
			n, radix_array, smem_size, threads_per_xform, xforms_per_block,
			self._params.min_mem_coalesce_width, self._params.num_smem_banks,
			self._params.size if normalize else 1,
			log2=log2, getPadding=getPadding, cuda=self._context.isCuda())


class GlobalFFTKernel(_FFTKernel):
	"""Generator for 'global' FFT kernel chain."""

	def __init__(self, fft_params, pass_num, n, curr_n, horiz_bs, dir, vert_bs, batch_size):
		_FFTKernel.__init__(self, fft_params)
		self._n = n
		self._curr_n = curr_n
		self._horiz_bs = horiz_bs
		self._dir = dir
		self._vert_bs = vert_bs
		self._starting_batch_size = batch_size
		self._pass_num = pass_num

	def generate(self, max_block_size, normalize):

		batch_size = self._starting_batch_size

		vertical = False if self._dir == X_DIRECTION else True

		radix_arr, radix1_arr, radix2_arr = getGlobalRadixInfo(self._n)

		num_passes = len(radix_arr)

		radix_init = self._horiz_bs if vertical else 1

		radix = radix_arr[self._pass_num]
		radix1 = radix1_arr[self._pass_num]
		radix2 = radix2_arr[self._pass_num]

		stride_in = radix_init
		for i in range(num_passes):
			if i != self._pass_num:
				stride_in *= radix_arr[i]

		stride_out = radix_init
		for i in range(self._pass_num):
			stride_out *= radix_arr[i]

		threads_per_xform = radix2
		batch_size = max_block_size if radix2 == 1 else batch_size
		batch_size = min(batch_size, stride_in)
		self._block_size = batch_size * threads_per_xform
		self._block_size = min(self._block_size, max_block_size)
		batch_size = self._block_size / threads_per_xform
		assert radix2 <= radix1
		assert radix1 * radix2 == radix
		assert radix1 <= self._params.max_radix

		numIter = radix1 / radix2

		blocks_per_xform = stride_in / batch_size
		num_blocks = blocks_per_xform
		if not vertical:
			num_blocks *= self._horiz_bs
		else:
			num_blocks *= self._vert_bs

		if radix2 == 1:
			smem_size = 0
		else:
			if stride_out == 1:
				smem_size = (radix + 1) * batch_size
			else:
				smem_size = self._block_size * radix1

		self._blocks_num = num_blocks

		if self._pass_num == num_passes - 1 and num_passes % 2 == 1:
			self.in_place_possible = True
		else:
			self.in_place_possible = False

		self._batch_size = batch_size

		return _template.get_def("globalKernel").render(
			self._params.scalar, self._params.complex, self._params.split, self._kernel_name,
			self._n, self._curr_n, self._pass_num,
			smem_size, batch_size,
			self._horiz_bs, self._vert_bs, vertical, max_block_size,
			self._params.size if normalize else 1,
			log2=log2, getGlobalRadixInfo=getGlobalRadixInfo, cuda=self._context.isCuda())

	def __get_batch_size(self):
		return self._batch_size

	batch_size = property(__get_batch_size)

	@staticmethod
	def createChain(fft_params, n, horiz_bs, dir, vert_bs):

		batch_size = fft_params.min_mem_coalesce_width
		vertical = not dir == X_DIRECTION

		radix_arr, radix1_arr, radix2_arr = getGlobalRadixInfo(n)

		num_passes = len(radix_arr)

		curr_n = n
		batch_size = min(horiz_bs, batch_size) if vertical else batch_size

		kernels = []

		for pass_num in range(num_passes):
			kernel = GlobalFFTKernel(fft_params, pass_num, n, curr_n, horiz_bs, dir, vert_bs, batch_size)
			kernel.compile(fft_params.max_block_size)
			batch_size = kernel.batch_size

			curr_n /= radix_arr[pass_num]

			kernels.append(kernel)

		return kernels
