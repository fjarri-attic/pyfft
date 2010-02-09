from pycuda.autoinit import device
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycudafft import *
import numpy

from cufft.pycuda_fft import *

import time
import math

MAX_BUFFER_SIZE = 16 # in megabytes

def log2(n):
	pos = 0
	for pow in [16, 8, 4, 2, 1]:
		if n >= 2 ** pow:
			n /= (2 ** pow)
			pos += pow
	return pos

def rand_complex(*dims):
	real = numpy.random.randn(*dims)
	imag = numpy.random.randn(*dims)
	return (real + 1j * imag).astype(numpy.complex64)
	#return (numpy.ones(dims) + 1j * numpy.ones(dims)).astype(numpy.complex64)

def difference(arr1, arr2, batch):
	#diff = numpy.abs(arr1 - arr2) / numpy.abs(arr1)
	#return diff.max()
	diff_arr = numpy.abs(arr1 - arr2).ravel()
	mod_arr = numpy.abs(arr1).ravel()

	max_diff = 0.0
	min_diff = 1.0e1000
	avg = 0
	problem_len = diff_arr.size / batch
	for i in range(batch):
		sum_diff = numpy.sum(diff_arr[i*problem_len:(i+1)*problem_len])
		sum_mod = numpy.sum(mod_arr[i*problem_len:(i+1)*problem_len+1])
		diff = math.sqrt(sum_diff / sum_mod)
		avg += diff
		max_diff = max_diff if diff < max_diff else diff
		min_diff = min_diff if diff > min_diff else diff

	return avg / batch

def numpy_fft_base(data, dim, len, batch, func):
	res = []
	for i in range(batch):
		if dim == clFFT_1D:
			part = data[i*len : (i+1)*len]
		elif dim == clFFT_2D:
			part = data[:, i*len : (i+1)*len]
		elif dim == clFFT_3D:
			part = data[:, :, i*len : (i+1)*len]

		x = func(part)
		res.append(x)

	return numpy.concatenate(tuple(res), axis=dim)

def getDim(x, y, z):
	if z == 1:
		if y == 1:
			return clFFT_1D
		else:
			return clFFT_2D
	else:
		return clFFT_3D

def getTestData(dim, x, y, z, batch):
	if dim == clFFT_1D:
		return rand_complex(x * batch)
	elif dim == clFFT_2D:
		return rand_complex(x, y * batch)
	elif dim == clFFT_3D:
		return rand_complex(x, y, z * batch)

def testPerformance(x, y, z):

	buf_size_bytes = MAX_BUFFER_SIZE * 1024 * 1024
	value_size = 8
	iterations = 10

	batch = buf_size_bytes / (x * y * z * value_size)

	if batch == 0:
		print "Buffer size is too big, skipping test"
		return

	dim = getDim(x, y, z)
	data = getTestData(dim, x, y, z, batch)

	a_gpu = gpuarray.to_gpu(data)
	b_gpu = gpuarray.GPUArray(data.shape, dtype=data.dtype)

	plan = FFTPlan(x, y, z, dim)

	start = cuda.Event()
	stop = cuda.Event()

	start.record()
	for i in xrange(iterations):
		clFFT_ExecuteInterleaved(plan, batch, clFFT_Forward, a_gpu.gpudata, b_gpu.gpudata)
	stop.record()
	stop.synchronize()
	t = stop.time_since(start) / 1000.0 # in seconds

	gflop = 5.0e-9 * (log2(x) + log2(y) + log2(z)) * x * y * z * batch

	print "* pycudafft performance: " + str([x, y, z]) + ", batch " + str(batch) + ": " + \
		str(t * 1000) + " ms, " + str(gflop / t) + " GFLOPS"

def testErrors(x, y, z, batch):

	buf_size_bytes = MAX_BUFFER_SIZE * 1024 * 1024
	value_size = 8
	large_epsilon = 1e-2 # for comparisons where errors are expected
	small_epsilon = 1e-7 # for comparisons where there shouldn't be any errors at all

	# Skip test if resulting data size is too big
	if x * y * z * batch * value_size > buf_size_bytes:
		#print "Array size is " + str(x * y * z * batch * value_size / 1024 / 1024) + " Mb - test skipped"
		return

	dim = getDim(x, y, z)
	data = getTestData(dim, x, y, z, batch)

	# Prepare arrays
	a_gpu = gpuarray.to_gpu(data)
	b_gpu = gpuarray.GPUArray(data.shape, dtype=data.dtype)

	# CUFFT tests
	cufft_plan = CUFFTPlan(x, y, z, batch)

	cufft_plan.execute(a_gpu, b_gpu, CUFFT_FORWARD)
	cufft_fw = b_gpu.get()

	cufft_plan.execute(b_gpu, a_gpu, CUFFT_INVERSE)
	cufft_res = a_gpu.get() / (x * y * z)

	cufft_err = difference(cufft_res, data, batch)

	del cufft_plan # forcefully release GPU memory

	# pycudafft tests
	plan = FFTPlan(x, y, z, dim)

	a_gpu.set(data)
	clFFT_ExecuteInterleaved(plan, batch, clFFT_Forward, a_gpu.gpudata, b_gpu.gpudata)
	pyfft_fw_outplace = b_gpu.get()

	clFFT_ExecuteInterleaved(plan, batch, clFFT_Inverse, b_gpu.gpudata, a_gpu.gpudata)
	pyfft_res_outplace = a_gpu.get() / (x * y * z)

	pycudafft_err_outplace = difference(pyfft_res_outplace, data, batch)

	a_gpu.set(data)
	clFFT_ExecuteInterleaved(plan, batch, clFFT_Forward, a_gpu.gpudata, a_gpu.gpudata)
	pyfft_fw_inplace = b_gpu.get()

	clFFT_ExecuteInterleaved(plan, batch, clFFT_Inverse, a_gpu.gpudata, a_gpu.gpudata)
	pyfft_res_inplace = a_gpu.get() / (x * y * z)

	pycudafft_err_inplace = difference(pyfft_res_inplace, data, batch)

	# check cases where there shouldn't be any errors at all
	pycudafft_err_inout_fw = difference(pyfft_fw_inplace, pyfft_fw_outplace, batch)
	pycudafft_err_inout_res = difference(pyfft_res_inplace, pyfft_res_outplace, batch)
	assert pycudafft_err_inout_fw < small_epsilon, "inplace-outplace intermediate error: " + str(pycudafft_err_inout_fw)
	assert pycudafft_err_inout_res < small_epsilon, "inplace-outplace final error: " + str(pycudafft_err_inout_res)

	# compare CUFFT and pycudafft results
	if cufft_err > large_epsilon:
		raise Exception("cufft forward-inverse error: " + str(cufft_err))

	if pycudafft_err_inplace > large_epsilon:
		raise Exception("pycudafft forward-inverse inplace error: " + str(pycudafft_err_inplace))

	if pycudafft_err_outplace > large_epsilon:
		raise Exception("pycudafft forward-inverse outplace error: " + str(pycudafft_err_outplace))

	diff_err = difference(cufft_fw, pyfft_fw_inplace, batch)
	if diff_err > large_epsilon:
		raise Exception("Difference between pycudafft and cufft: " + str(diff_err))

	print "* error tests for " + str([x, y, z]) + ", batch " + str(batch) + \
		": pycudafft=" + str(pycudafft_err_inplace) + \
		", cufft=" + str(cufft_err) + \
		", reference_check=" + str(diff_err)

def runErrorTests():
	for batch in [1, 16, 128, 1024, 4096]:

		# 1D
		for x in [8, 10, 13]:
			testErrors(2 ** x, 1, 1, batch)

		# 2D
		for x in [4, 7, 8, 10]:
			for y in [4, 7, 8, 10]:
				testErrors(2 ** x, 2 ** y, 1, batch)

		# 3D
		for x in [4, 7, 10]:
			for y in [4, 7, 10]:
				for z in [4, 7, 10]:
					testErrors(2 ** x, 2 ** y, 2 ** z, batch)

def runPerformanceTests():
	testPerformance(16, 1, 1)
	testPerformance(1024, 1, 1)
	testPerformance(8192, 1, 1)
	testPerformance(16, 16, 1)
	testPerformance(128, 128, 1)
	testPerformance(1024, 1024, 1)
	testPerformance(16, 16, 16)
	testPerformance(32, 32, 128)
	testPerformance(128, 128, 128)

#runErrorTests()
runPerformanceTests()
