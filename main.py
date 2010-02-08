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
	diff = numpy.abs(arr1 - arr2) / numpy.abs(arr1)
	return diff.max()

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
	epsilon = 1e-2

	if dim == clFFT_1D:
		data = rand_complex(x * batch)
	elif dim == clFFT_2D:
		data = rand_complex(x, y * batch)
	elif dim == clFFT_3D:
		data = rand_complex(x, y, z * batch)

	# Prepare arrays
	a_gpu = gpuarray.to_gpu(data)
	b_gpu = gpuarray.GPUArray(data.shape, dtype=data.dtype)

	# Prepare plans
	plan = FFTPlan(x, y, z, dim)
	cufft_plan = CUFFTPlan(x, y, z, batch)

	# CUFFT forward transform
	cufft_time = timefunc(cufft_plan.execute, a_gpu, b_gpu, -1)
	print "CUFFT Forward total call time: " + str(cufft_time) + " ms"
	cufft_fw = b_gpu.get()

	# CUFFT inverse transgorm
	cufft_time = timefunc(cufft_plan.execute, b_gpu, a_gpu, 1)
	cufft_res = a_gpu.get() / (x * y * z)

	# pycudafft forward transform
	a_gpu.set(data)
	pyfft_time_fw = timefunc(clFFT_ExecuteInterleaved, plan, batch, clFFT_Forward, a_gpu.gpudata, b_gpu.gpudata)
	print "PyCuda FFT Forward total call time: " + str(pyfft_time_fw) + " ms"
	pyfft_fw = b_gpu.get()

	# pycudafft inverse transform
	clFFT_ExecuteInterleaved(plan, batch, clFFT_Inverse, b_gpu.gpudata, a_gpu.gpudata)
	pyfft_res = a_gpu.get() / (x * y * z)

	# compare CUFFT and pycudafft results
	cufft_err = difference(cufft_res, data)
	print "cufft forward-inverse error: " + str(cufft_err)
	if cufft_err > epsilon:
		raise Exception("cufft forward-inverse error: " + str(cufft_err))

	pycudafft_err = difference(pyfft_res, data)
	print "pycudafft forward-inverse error: " + str(pycudafft_err)
	if pycudafft_err > epsilon:
		raise Exception("pycudafft forward-inverse error: " + str(pycudafft_err))

	diff_err = difference(cufft_fw, pyfft_fw)
	print "pycudafft - cufft error: " + str(diff_err)
	if diff_err > epsilon:
		raise Exception("Difference between pycudafft and cufft: " + str(diff_err))

def runTest(dim, x, y, z, batch):
	#try:
		print "--- Dim: " + str(dim + 1) + ", " + str([x, y, z]) + ", batch " + str(batch)
		test(dim, x, y, z, batch)
	#except Exception, e:
	#	print "Failed: " + str(e)

def runTests():
	for batch in [16, 128, 1024, 4096]:

		# 1D
		for x in [8, 10]:
			runTest(clFFT_1D, 2 ** x, 1, 1, batch)

		# 2D
		for x in [4, 7, 8, 10]:
			for y in [4, 7, 8, 10]:
				runTest(clFFT_2D, 2 ** x, 2 ** y, 1, batch)

		# 3D
		#for x in [1, 3, 7, 10]:
		#	for y in [1, 3, 7, 10]:
		#		for z in [1, 3, 7, 10]:
		#			runTest(clFFT_3D, 2 ** x, 2 ** y, 2 ** z, batch)

runTests()
