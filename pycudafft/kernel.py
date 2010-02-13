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

	radixArray = getRadixArray(n, 0)
	if n / radixArray[0] > plan.max_work_item_per_workgroup:
		radixArray = getRadixArray(n, plan.max_radix)

	assert radixArray[0] <= plan.max_radix, "max radix choosen is greater than allowed"
	assert n / radixArray[0] <= plan.max_work_item_per_workgroup, \
		"required work items per xform greater than maximum work items allowed per work group for local mem fft"

	kInfo = cl_fft_kernel_info()
	kInfo.num_workgroups = 0
	kInfo.num_workitems_per_workgroup = 0
	kInfo.dir = X_DIRECTION
	kInfo.in_place_possible = True
	kInfo.kernel_name = "fft"

	numWorkItemsPerXForm = n / radixArray[0]
	numWorkItemsPerWG = 64 if numWorkItemsPerXForm <= 64 else numWorkItemsPerXForm
	assert numWorkItemsPerWG <= plan.max_work_item_per_workgroup
	numXFormsPerWG = numWorkItemsPerWG / numWorkItemsPerXForm
	kInfo.num_workgroups = numXFormsPerWG
	kInfo.num_workitems_per_workgroup = numWorkItemsPerWG

	kInfo.lmem_size = getSharedMemorySize(n, radixArray, numWorkItemsPerXForm, numXFormsPerWG,
		plan.num_local_mem_banks, plan.min_mem_coalesce_width)

	kInfo.kernel_string = _template.get_def("localKernel").render(
		scalar='float',
		complex='float2',
		split=split,
		kernel_name=kInfo.kernel_name,
		shared_mem=kInfo.lmem_size,
		numWorkItemsPerXForm=numWorkItemsPerXForm,
		numXFormsPerWG=numXFormsPerWG,
		min_mem_coalesce_width=plan.min_mem_coalesce_width,
		N=radixArray,
		n=n,
		num_local_mem_banks=plan.num_local_mem_banks)

	return kInfo

def createGlobalFFTKernelString(plan, n, BS, dir, vertBS, split):

	maxThreadsPerBlock = plan.max_work_item_per_workgroup
	maxArrayLen = plan.max_radix
	batchSize = plan.min_mem_coalesce_width
	vertical = False if dir == X_DIRECTION else True

	radixArr, R1Arr, R2Arr = getGlobalRadixInfo(n)

	numPasses = len(radixArr)

	N = n
	Rinit = BS if vertical else 1
	batchSize = min(BS, batchSize) if vertical else batchSize

	kernels = []

	for passNum in range(numPasses):

		radix = radixArr[passNum]
		R1 = R1Arr[passNum]
		R2 = R2Arr[passNum]

		strideI = Rinit
		for i in range(numPasses):
			if i != passNum:
				strideI *= radixArr[i]

		strideO = Rinit
		for i in range(passNum):
			strideO *= radixArr[i]

		threadsPerXForm = R2
		batchSize = plan.max_work_item_per_workgroup if R2 == 1 else batchSize
		batchSize = min(batchSize, strideI)
		threadsPerBlock = batchSize * threadsPerXForm
		threadsPerBlock = min(threadsPerBlock, maxThreadsPerBlock)
		batchSize = threadsPerBlock / threadsPerXForm
		assert R2 <= R1
		assert R1*R2 == radix
		assert R1 <= maxArrayLen
		assert threadsPerBlock <= maxThreadsPerBlock

		numIter = R1 / R2

		numBlocksPerXForm = strideI / batchSize
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
				kInfo.lmem_size = (radix + 1) * batchSize
			else:
				kInfo.lmem_size = threadsPerBlock * R1

		kInfo.num_workgroups = numBlocks
		kInfo.num_workitems_per_workgroup = threadsPerBlock
		kInfo.dir = dir
		if passNum == numPasses - 1 and numPasses % 2 == 1:
			kInfo.in_place_possible = True
		else:
			kInfo.in_place_possible = False

		kInfo.kernel_name = kernelName

		kInfo.kernel_string = _template.get_def("globalKernel").render(
			scalar="float", complex="float2",
			split=split,
			passNum=passNum,
			kernel_name=kernelName,
			radixArr=radixArr,
			numPasses=numPasses,
			shared_mem=kInfo.lmem_size,
			R1Arr=R1Arr,
			R2Arr=R2Arr,
			Rinit=Rinit,
			batchSize=batchSize,
			BS=BS,
			vertBS=vertBS,
			vertical=vertical,
			maxThreadsPerBlock=maxThreadsPerBlock,
			max_work_item_per_workgroup=plan.max_work_item_per_workgroup,
			n=n,
			N=N
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
			radixArray = getRadixArray(plan.n.x, 0)
			if plan.n.x / radixArray[0] <= plan.max_work_item_per_workgroup:
				kernels.append(createLocalMemfftKernelString(plan, plan.split))
			else:
				radixArray = getRadixArray(plan.n.x, plan.max_radix)
				if plan.n.x / radixArray[0] <= plan.max_work_item_per_workgroup:
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
