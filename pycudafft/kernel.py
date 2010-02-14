import os.path

from mako.template import Template
from kernel_helpers import *

X_DIRECTION = 0
Y_DIRECTION = 1
Z_DIRECTION = 2

_dir, _file = os.path.split(os.path.abspath(__file__))
_template = Template(filename=os.path.join(_dir, 'kernel.mako'))

class cl_fft_kernel_info:
	def __init__(self):
		self.module = None
		self.kernel_name = ""
		self.smem_size = 0
		self.blocks_num = 0
		self.block_size = 0
		self.in_place_possible = None
		self.kernel_string = None

def createLocalMemfftKernelString(plan, split):

	n = plan.n.x
	assert n <= plan.max_block_size * plan.max_radix, "signal lenght too big for local mem fft"

	radix_array = getRadixArray(n, 0)
	if n / radix_array[0] > plan.max_block_size:
		radix_array = getRadixArray(n, plan.max_radix)

	assert radix_array[0] <= plan.max_radix, "max radix choosen is greater than allowed"
	assert n / radix_array[0] <= plan.max_block_size, \
		"required work items per xform greater than maximum work items allowed per work group for local mem fft"

	kinfo = cl_fft_kernel_info()
	kinfo.blocks_num = 0
	kinfo.block_size = 0
	kinfo.dir = X_DIRECTION
	kinfo.in_place_possible = True
	kinfo.kernel_name = "fft"

	threads_per_xform = n / radix_array[0]
	numWorkItemsPerWG = 64 if threads_per_xform <= 64 else threads_per_xform
	assert numWorkItemsPerWG <= plan.max_block_size
	xforms_per_block = numWorkItemsPerWG / threads_per_xform
	kinfo.blocks_num = xforms_per_block
	kinfo.block_size = numWorkItemsPerWG

	kinfo.smem_size = getSharedMemorySize(n, radix_array, threads_per_xform, xforms_per_block,
		plan.num_smem_banks, plan.min_mem_coalesce_width)

	kinfo.kernel_string = _template.get_def("localKernel").render(
		scalar='float',
		complex='float2',
		split=split,
		kernel_name=kinfo.kernel_name,
		shared_mem=kinfo.smem_size,
		threads_per_xform=threads_per_xform,
		xforms_per_block=xforms_per_block,
		min_mem_coalesce_width=plan.min_mem_coalesce_width,
		radix_arr=radix_array,
		n=n,
		num_smem_banks=plan.num_smem_banks,
		log2=log2,
		getPadding=getPadding)

	return kinfo

def createGlobalFFTKernelString(plan, n, horiz_bs, dir, vert_bs, split):

	max_block_size = plan.max_block_size
	max_array_len = plan.max_radix
	batch_size = plan.min_mem_coalesce_width
	vertical = False if dir == X_DIRECTION else True

	radix_arr, radix1_arr, radix2_arr = getGlobalRadixInfo(n)

	num_passes = len(radix_arr)

	curr_n = n
	radix_init = horiz_bs if vertical else 1
	batch_size = min(horiz_bs, batch_size) if vertical else batch_size

	kernels = []

	for pass_num in range(num_passes):

		radix = radix_arr[pass_num]
		radix1 = radix1_arr[pass_num]
		radix2 = radix2_arr[pass_num]

		stride_in = radix_init
		for i in range(num_passes):
			if i != pass_num:
				stride_in *= radix_arr[i]

		stride_out = radix_init
		for i in range(pass_num):
			stride_out *= radix_arr[i]

		threadsPerXForm = radix2
		batch_size = plan.max_block_size if radix2 == 1 else batch_size
		batch_size = min(batch_size, stride_in)
		block_size = batch_size * threadsPerXForm
		block_size = min(block_size, max_block_size)
		batch_size = block_size / threadsPerXForm
		assert radix2 <= radix1
		assert radix1 * radix2 == radix
		assert radix1 <= max_array_len
		assert block_size <= max_block_size

		numIter = radix1 / radix2

		blocks_per_xform = stride_in / batch_size
		num_blocks = blocks_per_xform
		if not vertical:
			num_blocks *= horiz_bs
		else:
			num_blocks *= vert_bs

		kernel_name = "fft"
		kinfo = cl_fft_kernel_info()
		kinfo.kernel = 0
		if radix2 == 1:
			kinfo.smem_size = 0
		else:
			if stride_out == 1:
				kinfo.smem_size = (radix + 1) * batch_size
			else:
				kinfo.smem_size = block_size * radix1

		kinfo.blocks_num = num_blocks
		kinfo.block_size = block_size
		kinfo.dir = dir
		if pass_num == num_passes - 1 and num_passes % 2 == 1:
			kinfo.in_place_possible = True
		else:
			kinfo.in_place_possible = False

		kinfo.kernel_name = kernel_name

		kinfo.kernel_string = _template.get_def("globalKernel").render(
			scalar="float", complex="float2",
			split=split,
			pass_num=pass_num,
			kernel_name=kernel_name,
			radix_arr=radix_arr,
			num_passes=num_passes,
			shared_mem=kinfo.smem_size,
			radix1_arr=radix1_arr,
			radix2_arr=radix2_arr,
			radix_init=radix_init,
			batch_size=batch_size,
			horiz_bs=horiz_bs,
			vert_bs=vert_bs,
			vertical=vertical,
			max_block_size=max_block_size,
			n=n,
			curr_n=curr_n,
			log2=log2,
			getPadding=getPadding
			)

		curr_n /= radix

		kernels.append(kinfo)

	return kernels

def FFT1D(plan, dir):

	kernels = []

	if dir == X_DIRECTION:
		if plan.n.x > plan.max_smem_fft_size:
			kernels.extend(createGlobalFFTKernelString(plan, plan.n.x, 1, X_DIRECTION, 1, plan.split))
		elif plan.n.x > 1:
			radix_array = getRadixArray(plan.n.x, 0)
			if plan.n.x / radix_array[0] <= plan.max_block_size:
				kernels.append(createLocalMemfftKernelString(plan, plan.split))
			else:
				radix_array = getRadixArray(plan.n.x, plan.max_radix)
				if plan.n.x / radix_array[0] <= plan.max_block_size:
					kernels.append(createLocalMemfftKernelString(plan, plan.split))
				else:
					# TODO: find out which conditions are necessary to execute this code
					kernels.extend(createGlobalFFTKernelString(plan, plan.n.x, 1, X_DIRECTION, 1, plan.split))
	elif dir == Y_DIRECTION:
		if plan.n.y > 1:
			kernels.extend(createGlobalFFTKernelString(plan, plan.n.y, plan.n.x, Y_DIRECTION, 1, plan.split))
	elif dir == Z_DIRECTION:
		if plan.n.z > 1:
			kernels.extend(createGlobalFFTKernelString(plan, plan.n.z, plan.n.x * plan.n.y, Z_DIRECTION, 1, plan.split))
	else:
		raise Exception("Wrong direction")

	return kernels
