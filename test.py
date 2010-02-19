from pycuda.autoinit import device
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy

from pycudafft import FFTPlan
from cufft import CUFFTPlan

import time
import math

FFT_1D = 0
FFT_2D = 1
FFT_3D = 2

MAX_BUFFER_SIZE = 16 # in megabytes

def log2(n):
	pos = 0
	for pow in [16, 8, 4, 2, 1]:
		if n >= 2 ** pow:
			n /= (2 ** pow)
			pos += pow
	return pos

def rand_real(*dims):
	return numpy.random.randn(*dims).astype(numpy.float32)
	#return numpy.ones(dims).astype(numpy.float32)

def rand_complex(*dims):
	real = rand_real(*dims)
	imag = rand_real(*dims)
	return (real + 1j * imag).astype(numpy.complex64)

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
		diff = sum_diff / sum_mod
		avg += diff
		max_diff = max_diff if diff < max_diff else diff
		min_diff = min_diff if diff > min_diff else diff

	return avg / batch

def numpy_fft_base(data, dim, len, batch, func):
	res = []
	for i in range(batch):
		if dim == FFT_1D:
			part = data[i*len : (i+1)*len]
		elif dim == FFT_2D:
			part = data[:, i*len : (i+1)*len]
		elif dim == FFT_3D:
			part = data[:, :, i*len : (i+1)*len]

		x = func(part)
		res.append(x)

	return numpy.concatenate(tuple(res), axis=dim)

def getDim(x, y, z):
	if z is None:
		if y is None:
			return FFT_1D
		else:
			return FFT_2D
	else:
		return FFT_3D

def getTestData(dim, x, y, z, batch, split):
	if dim == FFT_1D:
		dims = (x * batch,)
	elif dim == FFT_2D:
		dims = (x, y * batch)
	elif dim == FFT_3D:
		dims = (x, y, z * batch)

	if split:
		return rand_real(*dims), rand_real(*dims)
	else:
		return rand_complex(*dims)

def testPerformance(x, y=None, z=None):

	buf_size_bytes = MAX_BUFFER_SIZE * 1024 * 1024
	value_size = 8
	iterations = 10

	batch = buf_size_bytes / (x * (1 if y is None else y) *
		(1 if z is None else z) * value_size)

	if batch == 0:
		print "Buffer size is too big, skipping test"
		return

	dim = getDim(x, y, z)
	data = getTestData(dim, x, y, z, batch, False)

	a_gpu = gpuarray.to_gpu(data)
	b_gpu = gpuarray.GPUArray(data.shape, dtype=data.dtype)

	plan = FFTPlan(x, y, z)
	cufft_plan = CUFFTPlan(x, y, z, batch=batch)

	y = 1 if y is None else y
	z = 1 if z is None else z
	gflop = 5.0e-9 * (log2(x) + log2(y) + log2(z)) * x * y * z * batch

	start = cuda.Event()
	stop = cuda.Event()

	plan.execute(a_gpu, b_gpu, batch=batch) # warming up
	start.record()
	for i in xrange(iterations):
		plan.execute(a_gpu, b_gpu, batch=batch)
	stop.record()
	stop.synchronize()
	t_pycudafft = stop.time_since(start) / 1000.0 / iterations # in seconds

	cufft_plan.execute(a_gpu, b_gpu) # warming up
	start.record()
	for i in xrange(iterations):
		cufft_plan.execute(a_gpu, b_gpu)
	stop.record()
	stop.synchronize()
	t_cufft = stop.time_since(start) / 1000.0 / iterations # in seconds

	print "* pycudafft performance: " + str([x, y, z]) + ", batch " + str(batch) + ": " + \
		str(t_pycudafft * 1000) + " ms, " + str(gflop / t_pycudafft) + " GFLOPS"
	print "cufft: " + str(t_cufft * 1000) + " ms, " + str(gflop / t_cufft) + " GFLOPS"

def testErrors(x, y, z, batch, split):

	buf_size_bytes = MAX_BUFFER_SIZE * 1024 * 1024
	value_size = 8 # size of complex value, hardcoded (float, float)
	epsilon = 1e-6 # TODO: it depends on value type; 1e-6 is for float

	# Skip test if resulting data size is too big
	size = x * (1 if y is None else y) * (1 if z is None else z)
	if size * batch * value_size > buf_size_bytes:
		return

	dim = getDim(x, y, z)

	if split:
		data_re, data_im = getTestData(dim, x, y, z, batch, split)
		data = (data_re + 1j * data_im).astype(numpy.complex64)
	else:
		data = getTestData(dim, x, y, z, batch, split)

	# Prepare arrays
	a_gpu = gpuarray.to_gpu(data)
	b_gpu = gpuarray.GPUArray(data.shape, dtype=data.dtype)

	# CUFFT tests
	cufft_plan = CUFFTPlan(x, y, z, batch=batch)

	cufft_plan.execute(a_gpu, b_gpu)
	cufft_fw = b_gpu.get()

	cufft_plan.execute(b_gpu, a_gpu, inverse=True)
	cufft_res = a_gpu.get() / size

	cufft_err = difference(cufft_res, data, batch)

	# relese some GPU memory; this will help low-end videocards
	del cufft_plan
	if split:
		del a_gpu
		del b_gpu
		a_gpu_re = gpuarray.to_gpu(data_re)
		a_gpu_im = gpuarray.to_gpu(data_im)
		b_gpu_re = gpuarray.GPUArray(data_re.shape, dtype=data_re.dtype)
		b_gpu_im = gpuarray.GPUArray(data_im.shape, dtype=data_im.dtype)

	# pycudafft tests

	plan = FFTPlan(x, y, z, split=split)

	# out of place forward
	if split:
		a_gpu_re.set(data_re)
		a_gpu_im.set(data_im)
		plan.executeSplit(a_gpu_re, a_gpu_im,
			b_gpu_re, b_gpu_im, batch=batch)
		pyfft_fw_outplace = b_gpu_re.get() + 1j * b_gpu_im.get()
	else:
		a_gpu.set(data)
		plan.execute(a_gpu, b_gpu, batch=batch)
		pyfft_fw_outplace = b_gpu.get()

	# out of place inverse
	if split:
		plan.executeSplit(b_gpu_re, b_gpu_im,
			a_gpu_re, a_gpu_im, batch=batch, inverse=True)
		pyfft_res_outplace = (a_gpu_re.get() + 1j * a_gpu_im.get()) / size
	else:
		plan.execute(b_gpu, a_gpu, batch=batch, inverse=True)
		pyfft_res_outplace = a_gpu.get() / size

	pycudafft_err_outplace = difference(pyfft_res_outplace, data, batch)

	# inplace forward
	if split:
		a_gpu_re.set(data_re)
		a_gpu_im.set(data_im)
		plan.executeSplit(a_gpu_re, a_gpu_im, batch=batch)
		pyfft_fw_inplace = a_gpu_re.get() + 1j * a_gpu_im.get()
	else:
		a_gpu.set(data)
		plan.execute(a_gpu, batch=batch)
		pyfft_fw_inplace = a_gpu.get()

	# inplace inverse
	if split:
		plan.executeSplit(a_gpu_re, a_gpu_im, batch=batch, inverse=True)
		pyfft_res_inplace = (a_gpu_re.get() + 1j * a_gpu_im.get()) / size
	else:
		plan.execute(a_gpu, batch=batch, inverse=True)
		pyfft_res_inplace = a_gpu.get() / size

	pycudafft_err_inplace = difference(pyfft_res_inplace, data, batch)

	# check cases where there shouldn't be any errors at all
	pycudafft_err_inout_fw = difference(pyfft_fw_inplace, pyfft_fw_outplace, batch)
	pycudafft_err_inout_res = difference(pyfft_res_inplace, pyfft_res_outplace, batch)
	diff_err = difference(cufft_fw, pyfft_fw_inplace, batch)

	# compare CUFFT and pycudafft results
	assert pycudafft_err_inout_fw < epsilon, "inplace-outplace intermediate error: " + str(pycudafft_err_inout_fw)
	assert pycudafft_err_inout_res < epsilon, "inplace-outplace final error: " + str(pycudafft_err_inout_res)

	assert cufft_err < epsilon, "cufft forward-inverse error: " + str(cufft_err)
	assert pycudafft_err_inplace < epsilon, "pycudafft forward-inverse inplace error: " + str(pycudafft_err_inplace)
	assert pycudafft_err_outplace < epsilon, "pycudafft forward-inverse outplace error: " + str(pycudafft_err_outplace)

	assert diff_err < epsilon, "difference between pycudafft and cufft: " + str(diff_err)

def runErrorTests():

	def wrapper(x, y=None, z=None, batch=1, split=False):
		try:
			testErrors(x, y, z, batch=batch, split=split)
		except Exception, e:
			print "failed: " + str([x, y, z]) + ", batch " + str(batch) + \
				", " + ("split" if split else "interleaved") + \
				": " + str(e)

	for split in [False, True]:
		for batch in [1, 16, 128, 1024, 4096]:

			# 1D
			for x in [3, 8, 9, 10, 11, 13, 20]:
				wrapper(2 ** x, batch=batch, split=split)

			# 2D
			for x in [4, 7, 8, 10]:
				for y in [4, 7, 8, 10]:
					wrapper(2 ** x, 2 ** y, batch=batch, split=split)

			# 3D
			for x in [4, 7, 10]:
				for y in [4, 7, 10]:
					for z in [4, 7, 10]:
						wrapper(2 ** x, 2 ** y, 2 ** z, batch=batch, split=split)

	wrapper(16) # while plan.mem_coalesce_with = 32

def runPerformanceTests():
	testPerformance(16)
	testPerformance(1024)
	testPerformance(8192)
	testPerformance(16, 16)
	testPerformance(128, 128)
	testPerformance(1024, 1024)
	testPerformance(16, 16, 16)
	testPerformance(32, 32, 128)
	testPerformance(128, 128, 128)

runErrorTests()
#runPerformanceTests()
