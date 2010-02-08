from pycuda.autoinit import device
import pycuda.driver as cuda
from pycudafft import *
import numpy

import math

def rand_complex(*dims):
	#real = numpy.random.randn(*dims)
	#imag = numpy.random.randn(*dims)
	#return (real + 1j * imag).astype(numpy.complex64)

	return (numpy.ones(dims) + 1j * numpy.ones(dims)).astype(numpy.complex64)

def difference(arr1, arr2):
	return math.sqrt(numpy.cov(arr1.ravel() - arr2.ravel(), bias=1))

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

def test(dim, x, y, z, batch):

	if x * y * z * batch * 8 > 2 ** 26:
		print "Array size is " + str(x * y * z * batch * 8 / 1024 / 1024) + " Mb - test skipped"
		return

	if dim == clFFT_1D:
		data = rand_complex(x * batch)
		batch_len = x
		fft = numpy.fft.fft
		ifft = numpy.fft.ifft
	elif dim == clFFT_2D:
		data = rand_complex(x, y * batch)
		batch_len = y
		fft = numpy.fft.fft2
		ifft = numpy.fft.ifft2
	elif dim == clFFT_3D:
		data = rand_complex(x, y, z * batch)
		batch_len = z
		#fft = numpy.fft.fft3
		#ifft = numpy.fft.ifft3

	# TODO: Will not work under 32-bit python with 2D and 3D
	#numpy_transformed = numpy_fft_base(data, dim, batch_len, batch, fft)
	#numpy_back = numpy_fft_base(numpy_transformed, dim, batch_len, batch, ifft)
	#numpy_err = difference(numpy_back, data)
	#if numpy_err > 1e-6:
	#	raise Exception("numpy error: " + str(numpy_err))

	res = numpy.empty(data.shape).astype(numpy.complex64)

	plan = FFTPlan(x, y, z, dim)
	a_gpu = cuda.mem_alloc(data.nbytes)
	b_gpu = cuda.mem_alloc(data.nbytes)
	cuda.memcpy_htod(a_gpu, data)
	clFFT_ExecuteInterleaved(plan, batch, clFFT_Forward, a_gpu, b_gpu)

	#cuda.memcpy_dtoh(res, b_gpu)
	#print res

	clFFT_ExecuteInterleaved(plan, batch, clFFT_Forward, b_gpu, a_gpu)
	cuda.memcpy_dtoh(res, a_gpu)

	res = res / (x * y * z)

	#print data
	#print res

	cuda_err = difference(res, data)
	if cuda_err > 1e-6:
		raise Exception("cuda error: " + str(cuda_err))

def runTest(dim, x, y, z, batch):
#	try:
		print "--- Dim: " + str(dim + 1) + ", " + str([x, y, z]) + ", batch " + str(batch) + ":"
		test(dim, x, y, z, batch)
#	except Exception, e:
#		print "--- Dim: " + str(dim + 1) + ", " + str([x, y, z]) + ", batch " + str(batch) + ":"
#		print str(e)


def runTests():
	for batch in [1, 3, 4, 10]:

		# 1D
		for x in [8, 10]:
			runTest(clFFT_1D, 2 ** x, 1, 1, batch)

		# 2D
		for x in [1, 3, 4, 7, 8, 10]:
			for y in [1, 3, 4, 7, 8, 10]:
				runTest(clFFT_2D, 2 ** x, 2 ** y, 1, batch)

		# 3D
		for x in [1, 3, 7, 10]:
			for y in [1, 3, 7, 10]:
				for z in [1, 3, 7, 10]:
					runTest(clFFT_3D, 2 ** x, 2 ** y, 2 ** z, batch)

runTests()
