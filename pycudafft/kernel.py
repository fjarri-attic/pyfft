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
		self.lmem_size = 0
		self.num_workgroups = 0
		self.num_workitems_per_workgroup = 0
		self.cl_fft_kernel_dir = None
		self.in_place_possible = None
		self.kernel_string = None

def createLocalMemfftKernelString(plan, split):

	n = plan.n.x
	assert n <= plan.max_work_item_per_workgroup * plan.max_radix, "signal lenght too big for local mem fft"

	radix_array = getRadixArray(n, 0)
	if n / radix_array[0] > plan.max_work_item_per_workgroup:
		radix_array = getRadixArray(n, plan.max_radix)

	assert radix_array[0] <= plan.max_radix, "max radix choosen is greater than allowed"
	assert n / radix_array[0] <= plan.max_work_item_per_workgroup, \
		"required work items per xform greater than maximum work items allowed per work group for local mem fft"

	kInfo = cl_fft_kernel_info()
	kInfo.num_workgroups = 0
	kInfo.num_workitems_per_workgroup = 0
	kInfo.dir = X_DIRECTION
	kInfo.in_place_possible = True
	kInfo.kernel_name = "fft"

	threads_per_xform = n / radix_array[0]
	numWorkItemsPerWG = 64 if threads_per_xform <= 64 else threads_per_xform
	assert numWorkItemsPerWG <= plan.max_work_item_per_workgroup
	xforms_per_block = numWorkItemsPerWG / threads_per_xform
	kInfo.num_workgroups = xforms_per_block
	kInfo.num_workitems_per_workgroup = numWorkItemsPerWG

	kInfo.lmem_size = getSharedMemorySize(n, radix_array, threads_per_xform, xforms_per_block,
		plan.num_local_mem_banks, plan.min_mem_coalesce_width)

	kInfo.kernel_string = _template.get_def("localKernel").render(
		scalar='float',
		complex='float2',
		split=split,
		kernel_name=kInfo.kernel_name,
		shared_mem=kInfo.lmem_size,
		threads_per_xform=threads_per_xform,
		xforms_per_block=xforms_per_block,
		min_mem_coalesce_width=plan.min_mem_coalesce_width,
		radix_arr=radix_array,
		n=n,
		num_local_mem_banks=plan.num_local_mem_banks,
		log2=log2,
		getPadding=getPadding)

	return kInfo

def createGlobalFFTKernelString(plan, n, BS, dir, vertBS, split):

	max_block_size = plan.max_work_item_per_workgroup
	maxArrayLen = plan.max_radix
	batch_size = plan.min_mem_coalesce_width
	vertical = False if dir == X_DIRECTION else True

	radix_arr, R1Arr, R2Arr = getGlobalRadixInfo(n)

	num_passes = len(radix_arr)

	N = n
	Rinit = BS if vertical else 1
	batch_size = min(BS, batch_size) if vertical else batch_size

	kernels = []

	for pass_num in range(num_passes):

		radix = radix_arr[pass_num]
		R1 = R1Arr[pass_num]
		R2 = R2Arr[pass_num]

		strideI = Rinit
		for i in range(num_passes):
			if i != pass_num:
				strideI *= radix_arr[i]

		strideO = Rinit
		for i in range(pass_num):
			strideO *= radix_arr[i]

		threadsPerXForm = R2
		batch_size = plan.max_work_item_per_workgroup if R2 == 1 else batch_size
		batch_size = min(batch_size, strideI)
		block_size = batch_size * threadsPerXForm
		block_size = min(block_size, max_block_size)
		batch_size = block_size / threadsPerXForm
		assert R2 <= R1
		assert R1*R2 == radix
		assert R1 <= maxArrayLen
		assert block_size <= max_block_size

		numIter = R1 / R2

		numBlocksPerXForm = strideI / batch_size
		numBlocks = numBlocksPerXForm
		if not vertical:
			numBlocks *= BS
		else:
			numBlocks *= vertBS

		kernelName = "fft"
		kInfo = cl_fft_kernel_info()
		kInfo.kernel = 0
		if R2 == 1:
			kInfo.lmem_size = 0
		else:
			if strideO == 1:
				kInfo.lmem_size = (radix + 1) * batch_size
			else:
				kInfo.lmem_size = block_size * R1

		kInfo.num_workgroups = numBlocks
		kInfo.num_workitems_per_workgroup = block_size
		kInfo.dir = dir
		if pass_num == num_passes - 1 and num_passes % 2 == 1:
			kInfo.in_place_possible = True
		else:
			kInfo.in_place_possible = False

		kInfo.kernel_name = kernelName

		kInfo.kernel_string = _template.get_def("globalKernel").render(
			scalar="float", complex="float2",
			split=split,
			pass_num=pass_num,
			kernel_name=kernelName,
			radix_arr=radix_arr,
			num_passes=num_passes,
			shared_mem=kInfo.lmem_size,
			R1Arr=R1Arr,
			R2Arr=R2Arr,
			Rinit=Rinit,
			batch_size=batch_size,
			BS=BS,
			vertBS=vertBS,
			vertical=vertical,
			max_block_size=max_block_size,
			n=n,
			N=N,
			log2=log2,
			getPadding=getPadding
			)

		N /= radix

		kernels.append(kInfo)

	return kernels

def FFT1D(plan, dir):

	kernels = []

	if dir == X_DIRECTION:
		if plan.n.x > plan.max_localmem_fft_size:
			kernels.extend(createGlobalFFTKernelString(plan, plan.n.x, 1, X_DIRECTION, 1, plan.split))
		elif plan.n.x > 1:
			radix_array = getRadixArray(plan.n.x, 0)
			if plan.n.x / radix_array[0] <= plan.max_work_item_per_workgroup:
				kernels.append(createLocalMemfftKernelString(plan, plan.split))
			else:
				radix_array = getRadixArray(plan.n.x, plan.max_radix)
				if plan.n.x / radix_array[0] <= plan.max_work_item_per_workgroup:
					kernels.append(createLocalMemfftKernelString(plan, plan.split))
				else:
					# TODO: find out which conditions are necessary to execute this code
					kernels.extend(createGlobalFFTKernelString(plan, plan.n.x, 1, X_DIRECTION, 1, plan.split))
	elif dir == Y_DIRECTION:
		if plan.n.y > 1:
			kernels.extend(createGlobalFFTKernelString(plan, plan.n.y, plan.n.x, Y_DIRECTION, 1, plan.split))
	elif dir == Z_DIRECTION:
		if plan.n.z > 1:
			kernels.extend(createGlobalFFTKernelString(plan, plan.n.z, plan.n.x*plan.n.y, Z_DIRECTION, 1, plan.split))
	else:
		raise Exception("Wrong direction")

	return kernels
