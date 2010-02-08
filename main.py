from pycuda.autoinit import device
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycudafft import *
import numpy

from cufft.pycuda_fft import *


import time
import math

def rand_complex(*dims):
	real = numpy.random.randn(*dims)
	imag = numpy.random.randn(*dims)
	return (real + 1j * imag).astype(numpy.complex64)
	#return (numpy.ones(dims) + 1j * numpy.ones(dims)).astype(numpy.complex64)

def difference(arr1, arr2):
	return math.sqrt(numpy.cov(arr1.ravel() - arr2.ravel(), bias=1)) / arr1.size

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

def timefunc(func, *args, **kwds):
	t1 = time.time()
	func(*args, **kwds)
	return (time.time() - t1) * 1000

def test(dim, x, y, z, batch):

	if x * y * z * batch * 8 > 2 ** 26:
		print "Array size is " + str(x * y * z * batch * 8 / 1024 / 1024) + " Mb - test skipped"
		return

	if dim == clFFT_1D:
		data = rand_complex(x * batch)
		cufft_plan = get_1dplan((x,), batch)
	elif dim == clFFT_2D:
		data = rand_complex(x, y * batch)
	elif dim == clFFT_3D:
		data = rand_complex(x, y, z * batch)

	# Prepare arrays
	res = numpy.empty(data.shape, dtype=numpy.complex64)
	cufft_res = numpy.empty(data.shape, dtype=numpy.complex64)
	pyfft_res = numpy.empty(data.shape, dtype=numpy.complex64)

	a_gpu = gpuarray.to_gpu(data)
	b_gpu = gpuarray.GPUArray(data.shape, dtype=data.dtype)

	# Prepare pycudaftt plan
	plan = FFTPlan(x, y, z, dim)

	# CUFFT forward transform
	cufft_time = timefunc(gpu_fft, a_gpu, b_gpu, plan=cufft_plan)
	print "CUFFT Forward total call time: " + str(cufft_time) + " ms"
	cufft_res = b_gpu.get()

	# pycudafft forward transform
	a_gpu.set(data)
	pyfft_time_fw = timefunc(clFFT_ExecuteInterleaved, plan, batch, clFFT_Forward, a_gpu.gpudata, b_gpu.gpudata)
	print "PyCuda FFT Forward total call time: " + str(pyfft_time_fw) + " ms"
	pyfft_res = b_gpu.get()

	# compare CUFFT and pycudafft results
	diff_err = difference(cufft_res, pyfft_res)
	if diff_err > 1e-6:
		raise Exception("Difference between pycudafft and cufft: " + str(diff_err))

	# pycudafft inerse transform
	clFFT_ExecuteInterleaved(plan, batch, clFFT_Inverse, b_gpu.gpudata, a_gpu.gpudata)
	res = a_gpu.get()
	res = res / (x * y * z)

	# compare forward-inverse result with initial data
	pycudafft_err = difference(res, data)
	if pycudafft_err > 1e-6:
		raise Exception("pycudafft forward-inverse error: " + str(pycudafft_err))

def runTest(dim, x, y, z, batch):
	#try:
		print "--- Dim: " + str(dim + 1) + ", " + str([x, y, z]) + ", batch " + str(batch)
		test(dim, x, y, z, batch)
	#except Exception, e:
	#	print "Failed: " + str(e)

def runTests():
	for batch in [10, 100, 1000, 5000]:

		# 1D
		for x in [8, 10]:
			runTest(clFFT_1D, 2 ** x, 1, 1, batch)

		# 2D
		#for x in [1, 3, 4, 7, 8, 10]:
		#	for y in [1, 3, 4, 7, 8, 10]:
		#		runTest(clFFT_2D, 2 ** x, 2 ** y, 1, batch)

		# 3D
		#for x in [1, 3, 7, 10]:
		#	for y in [1, 3, 7, 10]:
		#		for z in [1, 3, 7, 10]:
		#			runTest(clFFT_3D, 2 ** x, 2 ** y, 2 ** z, batch)

runTests()
