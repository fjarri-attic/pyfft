import os.path

from mako.template import Template
from pycuda.compiler import SourceModule

from .kernel_helpers import *

X_DIRECTION = 0
Y_DIRECTION = 1
Z_DIRECTION = 2

_dir, _file = os.path.split(os.path.abspath(__file__))
_template = Template(filename=os.path.join(_dir, 'kernel.mako'))

class _FFTKernel:

	def __init__(self, plan):
		self.x = plan.n.x
		self.y = plan.n.y
		self.z = plan.n.z
		self.num_smem_banks = plan.num_smem_banks
		self.min_mem_coalesce_width = plan.min_mem_coalesce_width
		self.max_radix = plan.max_radix
		self.kernel_name = "fft"
		self.max_registers = plan.max_registers
		self.max_smem_fft_size = plan.max_smem_fft_size
		self.split = plan.split
		self.max_grid_x = plan.max_grid_x

		self.previous_batch = None

		self.scalar = plan.scalar
		self.complex = plan.complex

	def compile(self, max_block_size):
		self.module = None
		self.func_ref = None
		max_block_size = max_block_size * 2

		while max_block_size > self.num_smem_banks:
			max_block_size /= 2

			try:
				kernel_string = self.generate(max_block_size)
				self.kernel_string = kernel_string
			except AssertionError as e:
				continue

			module = SourceModule(kernel_string, no_extern_c=True)
			func_ref_forward = module.get_function(self.kernel_name + "_forward")
			func_ref_inverse = module.get_function(self.kernel_name + "_inverse")

			if func_ref_forward.num_regs * self.block_size > self.max_registers:
				continue

			self.module = module
			self.func_ref_forward = func_ref_forward
			self.func_ref_inverse = func_ref_inverse
			break

		if self.module is None:
			raise Exception("Failed to find block size for the kernel")

	def prepare(self, batch):
		if self.previous_batch != batch:
			self.previous_batch = batch
			batch, grid = self._getKernelWorkDimensions(batch)
			self.exec_grid = grid
			self.exec_batch = batch

			if self.split:
				self.func_ref_forward.prepare("PPPPi", block=(self.block_size, 1, 1))
				self.func_ref_inverse.prepare("PPPPi", block=(self.block_size, 1, 1))
			else:
				self.func_ref_forward.prepare("PPi", block=(self.block_size, 1, 1))
				self.func_ref_inverse.prepare("PPi", block=(self.block_size, 1, 1))

	def preparedCall(self, data_in, data_out, inverse):
		if inverse:
			func_ref = self.func_ref_inverse
		else:
			func_ref = self.func_ref_forward

		func_ref.prepared_call(self.exec_grid, data_in, data_out, self.exec_batch)

	def preparedCallSplit(self, data_in_re, data_in_im, data_out_re, data_out_im, inverse):
		if inverse:
			func_ref = self.func_ref_inverse
		else:
			func_ref = self.func_ref_forward

		func_ref.prepared_call(self.exec_grid, data_in_re, data_in_im, data_out_re,
			data_out_im, self.exec_batch)

	def _getKernelWorkDimensions(self, batch):
		blocks_num = self.blocks_num

		if self.dir == X_DIRECTION:
			if self.x <= self.max_smem_fft_size:
				batch = self.y * self.z * batch
				blocks_num = (batch / blocks_num + 1) if batch % blocks_num != 0 else batch / blocks_num
			else:
				batch *= self.y * self.z
				blocks_num *= batch
		elif self.dir == Y_DIRECTION:
			batch *= self.z
			blocks_num *= batch
		elif self.dir == Z_DIRECTION:
			blocks_num *= batch

		if blocks_num > self.max_grid_x:
			grid = (self.max_grid_x, self.max_grid_x / blocks_num)
		else:
			grid = (blocks_num, 1)

		return batch, grid


class LocalFFTKernel(_FFTKernel):

	def __init__(self, plan, n):
		_FFTKernel.__init__(self, plan)
		self.n = n

	def generate(self, max_block_size):
		n = self.n
		assert n <= max_block_size * self.max_radix, "signal lenght too big for local mem fft"

		radix_array = getRadixArray(n, 0)
		if n / radix_array[0] > max_block_size:
			radix_array = getRadixArray(n, self.max_radix)

		assert radix_array[0] <= self.max_radix, "max radix choosen is greater than allowed"
		assert n / radix_array[0] <= max_block_size, \
			"required work items per xform greater than maximum work items allowed per work group for local mem fft"

		self.dir = X_DIRECTION
		self.in_place_possible = True

		threads_per_xform = n / radix_array[0]
		block_size = 64 if threads_per_xform <= 64 else threads_per_xform
		assert block_size <= max_block_size
		xforms_per_block = block_size / threads_per_xform
		self.blocks_num = xforms_per_block
		self.block_size = block_size

		smem_size = getSharedMemorySize(n, radix_array, threads_per_xform, xforms_per_block,
			self.num_smem_banks, self.min_mem_coalesce_width)

		return _template.get_def("localKernel").render(
			self.scalar, self.complex, self.split, self.kernel_name,
			n, radix_array, smem_size, threads_per_xform, xforms_per_block,
			self.min_mem_coalesce_width, self.num_smem_banks,
			log2=log2, getPadding=getPadding)


class GlobalFFTKernel(_FFTKernel):

	def __init__(self, plan, pass_num, n, curr_n, horiz_bs, dir, vert_bs, batch_size):
		_FFTKernel.__init__(self, plan)
		self.n = n
		self.curr_n = curr_n
		self.horiz_bs = horiz_bs
		self.dir = dir
		self.vert_bs = vert_bs
		self.batch_size = batch_size
		self.pass_num = pass_num

	def generate(self, max_block_size):

		batch_size = self.batch_size

		vertical = False if self.dir == X_DIRECTION else True

		radix_arr, radix1_arr, radix2_arr = getGlobalRadixInfo(self.n)

		num_passes = len(radix_arr)

		radix_init = self.horiz_bs if vertical else 1

		radix = radix_arr[self.pass_num]
		radix1 = radix1_arr[self.pass_num]
		radix2 = radix2_arr[self.pass_num]

		stride_in = radix_init
		for i in range(num_passes):
			if i != self.pass_num:
				stride_in *= radix_arr[i]

		stride_out = radix_init
		for i in range(self.pass_num):
			stride_out *= radix_arr[i]

		threads_per_xform = radix2
		batch_size = max_block_size if radix2 == 1 else batch_size
		batch_size = min(batch_size, stride_in)
		self.block_size = batch_size * threads_per_xform
		self.block_size = min(self.block_size, max_block_size)
		batch_size = self.block_size / threads_per_xform
		assert radix2 <= radix1
		assert radix1 * radix2 == radix
		assert radix1 <= self.max_radix

		numIter = radix1 / radix2

		blocks_per_xform = stride_in / batch_size
		num_blocks = blocks_per_xform
		if not vertical:
			num_blocks *= self.horiz_bs
		else:
			num_blocks *= self.vert_bs

		if radix2 == 1:
			self.smem_size = 0
		else:
			if stride_out == 1:
				self.smem_size = (radix + 1) * batch_size
			else:
				self.smem_size = self.block_size * radix1

		self.blocks_num = num_blocks

		if self.pass_num == num_passes - 1 and num_passes % 2 == 1:
			self.in_place_possible = True
		else:
			self.in_place_possible = False

		self.calculated_batch_size = batch_size

		return _template.get_def("globalKernel").render(
			self.scalar, self.complex, self.split, self.kernel_name,
			self.n, self.curr_n, self.pass_num,
			self.smem_size, batch_size,
			self.horiz_bs, self.vert_bs, vertical, max_block_size,
			log2=log2, getGlobalRadixInfo=getGlobalRadixInfo)

	@staticmethod
	def createChain(plan, n, horiz_bs, dir, vert_bs):

		batch_size = plan.min_mem_coalesce_width
		vertical = not dir == X_DIRECTION

		radix_arr, radix1_arr, radix2_arr = getGlobalRadixInfo(n)

		num_passes = len(radix_arr)

		curr_n = n
		batch_size = min(horiz_bs, batch_size) if vertical else batch_size

		kernels = []

		for pass_num in range(num_passes):
			kernel = GlobalFFTKernel(plan, pass_num, n, curr_n, horiz_bs, dir, vert_bs, batch_size)
			kernel.compile(plan.max_block_size)
			batch_size = kernel.calculated_batch_size

			curr_n /= radix_arr[pass_num]

			kernels.append(kernel)

		return kernels
