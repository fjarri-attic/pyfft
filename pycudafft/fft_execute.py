from fft_internal import *
import pycuda.driver as cuda
import numpy

def getKernelWorkDimensions(plan, kernelInfo, batchSize):

	lWorkItems = kernelInfo.num_workitems_per_workgroup
	numWorkGroups = kernelInfo.num_workgroups

	if kernelInfo.dir == cl_fft_kernel_x:
		maxLocalMemFFTSize = plan.max_localmem_fft_size
		if plan.n.x <= maxLocalMemFFTSize:
			batchSize = plan.n.y * plan.n.z * batchSize
			numWorkGroups = (batchSize / numWorkGroups + 1) if batchSize % numWorkGroups != 0 else batchSize / numWorkGroups
		else:
			batchSize *= plan.n.y * plan.n.z
			numWorkGroups *= batchSize
	elif kernelInfo.dir == cl_fft_kernel_y:
		batchSize *= plan.n.z
		numWorkGroups *= batchSize
	elif kernelInfo.dir == cl_fft_kernel_z:
		numWorkGroups *= batchSize

	gWorkItems = numWorkGroups * lWorkItems
	return batchSize, gWorkItems, lWorkItems

def clFFT_ExecuteInterleaved(plan, batchSize, dir, data_in, data_out):

	inPlaceDone = 0
	isInPlace = (data_in == data_out)

	if plan.temp_buffer_needed and plan.last_batch_size != batchSize:
		plan.last_batch_size = batchSize
		# TODO: remove hardcoded '2 * 4' when adding support for different types
		plan.tempmemobj = cuda.mem_alloc(plan.n.x * plan.n.y * plan.n.z * batchSize * 2 * 4)

	memObj = (data_in, data_out, plan.tempmemobj)
	kernelInfo = plan.kernel_info
	numKernels = len(plan.kernel_info)

	numKernelsOdd = (numKernels % 2 == 1)
	currRead  = 0
	currWrite = 1

	# at least one external dram shuffle (transpose) required
	inPlaceDone = False
	if plan.temp_buffer_needed:
		# in-place transform
		if isInPlace:
			currRead  = 1
			currWrite = 2
		else:
			currWrite = 1 if numKernelsOdd else 2

		for kInfo in kernelInfo:
			if isInPlace and numKernelsOdd and not inPlaceDone and kInfo.in_place_possible:
				currWrite = currRead
				inPlaceDone = True

			s = batchSize
			s, gWorkItems, lWorkItems = getKernelWorkDimensions(plan, kInfo, s)

			func = kInfo.function_ref
			# TODO: remove hardcoded '4' (should be value type size)
			# TODO: prepare functions when creating the plan
			#print "Launching " + kInfo.kernel_name + ": grid " + str(gWorkItems / lWorkItems) + \
			#	", block " + str(lWorkItems) + ", registers " + str(func.num_regs) + \
			#	", local mem " + str(func.local_size_bytes) + ", shared mem " + str(func.shared_size_bytes) + \
			#	" (kInfo: " + str(kInfo.lmem_size * 4) + ")"
			func.prepare("PPii", block=(lWorkItems, 1, 1)) #, shared=4 * kInfo.lmem_size)
			func.prepared_call((gWorkItems / lWorkItems, 1), memObj[currRead], memObj[currWrite], dir, s)

			currRead  = 1 if (currWrite == 1) else 2
			currWrite = 2 if (currWrite == 1) else 1

	# no dram shuffle (transpose required) transform
	# all kernels can execute in-place.
	else:
		for kInfo in kernelInfo:

			s = batchSize
			s, gWorkItems, lWorkItems = getKernelWorkDimensions(plan, kInfo, s)

			func = kInfo.function_ref
			# TODO: remove hardcoded '4' (should be value type size)
			# TODO: prepare functions when creating the plan
			#print "Launching " + kInfo.kernel_name + ": grid " + str(gWorkItems / lWorkItems) + \
			#	", block " + str(lWorkItems) + ", registers " + str(func.num_regs) + \
			#	", local mem " + str(func.local_size_bytes) + ", shared mem " + str(func.shared_size_bytes) + \
			#	" (kInfo: " + str(kInfo.lmem_size * 4) + ")"
			func.prepare("PPii", block=(lWorkItems, 1, 1)) #, shared=4 * kInfo.lmem_size)
			func.prepared_call((gWorkItems / lWorkItems, 1), memObj[currRead], memObj[currWrite], dir, s)

			currRead  = 1
			currWrite = 1
