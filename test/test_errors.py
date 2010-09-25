import numpy

from helpers import *

def numpyFFT(func, data, batch):
	res = numpy.empty(data.size, dtype=data.dtype)
	data_flat = data.ravel()
	size = data.size / batch

	single_shape = list(data.shape)
	single_shape[0] /= batch
	single_shape = tuple(single_shape)

	for i in xrange(batch):
		res[i*size:(i+1)*size] = func(data_flat[i*size:(i+1)*size].reshape(single_shape)).ravel()
	return res.reshape(data.shape)

def testErrors(ctx, shape, dtype, batch, fast_math):

	if dtype in DOUBLE_DTYPES:
		epsilon = 1e-11
	else:
		epsilon = 1.1e-6

	split = (dtype not in COMPLEX_DTYPES)
	complex_dtype = numpy.complex128 if dtype in DOUBLE_DTYPES else numpy.complex64

	if split:
		data_re, data_im = getTestData(shape, dtype, batch)
		data = (data_re + 1j * data_im).astype(complex_dtype)
	else:
		data = getTestData(shape, dtype, batch)

	# Reference tests
	numpy_fw = numpyFFT(numpy.fft.fftn, data, batch)
	numpy_res = numpyFFT(numpy.fft.ifftn, numpy_fw, batch)

	numpy_err = difference(numpy_res, data, batch)

	# Prepare arrays
	if split:
		a_gpu_re = ctx.toGpu(data_re)
		a_gpu_im = ctx.toGpu(data_im)
		b_gpu_re = ctx.allocate(data_re.shape, data_re.dtype)
		b_gpu_im = ctx.allocate(data_im.shape, data_im.dtype)
	else:
		a_gpu = ctx.toGpu(data)
		b_gpu = ctx.allocate(data.shape, data.dtype)

	# pyfft tests

	plan = ctx.getPlan(shape, dtype=dtype, context=ctx.context, normalize=True,
		wait_for_finish=True, fast_math=fast_math)

	# out of place forward
	if split:
		plan.execute(a_gpu_re, a_gpu_im, b_gpu_re, b_gpu_im, batch=batch)
		pyfft_fw_outplace = ctx.fromGpu(b_gpu_re, data_re.shape, data_re.dtype) + \
			1j * ctx.fromGpu(b_gpu_im, data_im.shape, data_im.dtype)
	else:
		plan.execute(a_gpu, b_gpu, batch=batch)
		pyfft_fw_outplace = ctx.fromGpu(b_gpu, data.shape, data.dtype)

	# out of place inverse
	if split:
		plan.execute(b_gpu_re, b_gpu_im,
			a_gpu_re, a_gpu_im, batch=batch, inverse=True)
		pyfft_res_outplace = ctx.fromGpu(a_gpu_re, data_re.shape, data_re.dtype) + \
			1j * ctx.fromGpu(a_gpu_im, data_im.shape, data_im.dtype)
	else:
		plan.execute(b_gpu, a_gpu, batch=batch, inverse=True)
		pyfft_res_outplace = ctx.fromGpu(a_gpu, data.shape, data.dtype)

	pyfft_err_outplace = difference(pyfft_res_outplace, data, batch)

	# inplace forward
	if split:
		a_gpu_re = ctx.toGpu(data_re)
		a_gpu_im = ctx.toGpu(data_im)
		plan.execute(a_gpu_re, a_gpu_im, batch=batch)
		pyfft_fw_inplace = ctx.fromGpu(a_gpu_re, data_re.shape, data_re.dtype) + \
			1j * ctx.fromGpu(a_gpu_im, data_im.shape, data_im.dtype)
	else:
		a_gpu = ctx.toGpu(data)
		plan.execute(a_gpu, batch=batch)
		pyfft_fw_inplace = ctx.fromGpu(a_gpu, data.shape, data.dtype)

	# inplace inverse
	if split:
		plan.execute(a_gpu_re, a_gpu_im, batch=batch, inverse=True)
		pyfft_res_inplace = ctx.fromGpu(a_gpu_re, data_re.shape, data_re.dtype) + \
			1j * ctx.fromGpu(a_gpu_im, data_im.shape, data_im.dtype)
	else:
		plan.execute(a_gpu, batch=batch, inverse=True)
		pyfft_res_inplace = ctx.fromGpu(a_gpu, data.shape, data.dtype)

	pyfft_err_inplace = difference(pyfft_res_inplace, data, batch)

	# check cases where there shouldn't be any errors at all
	pyfft_err_inout_fw = difference(pyfft_fw_inplace, pyfft_fw_outplace, batch)
	pyfft_err_inout_res = difference(pyfft_res_inplace, pyfft_res_outplace, batch)
	diff_err = difference(numpy_fw, pyfft_fw_inplace, batch)

	# compare numpy and pyfft results
	assert pyfft_err_inout_fw < epsilon, "inplace-outplace intermediate error: " + str(pyfft_err_inout_fw)
	assert pyfft_err_inout_res < epsilon, "inplace-outplace final error: " + str(pyfft_err_inout_res)

	assert numpy_err < epsilon, "numpy forward-inverse error: " + str(numpy_err)
	assert pyfft_err_inplace < epsilon, "pyfft forward-inverse inplace error: " + str(pyfft_err_inplace)
	assert pyfft_err_outplace < epsilon, "pyfft forward-inverse outplace error: " + str(pyfft_err_outplace)

	assert diff_err < epsilon, "difference between pyfft and numpy: " + str(diff_err)

	return pyfft_err_inplace, diff_err

def run(test_cuda, test_opencl, buffer_size, fast_math, double):
	print "Running error tests" + \
		(", double precision" if double else ", single precision") + \
		(", fast math" if fast_math else ", accurate math") + "..."

	# Fill shapes
	shapes = []

	# 1D
	for x in [3, 8, 9, 10, 11, 13, 20]:
		shapes.append((2 ** x,))

	# 2D
	for x in [4, 7, 8, 10]:
		for y in [4, 7, 8, 10]:
			shapes.append((2 ** x, 2 ** y))

	# 3D
	for x in [4, 7, 10]:
		for y in [4, 7, 10]:
			for z in [4, 7, 10]:
				shapes.append((2 ** x, 2 ** y, 2 ** z))

	batch_sizes = [1, 16, 128, 1024, 4096]
	dtypes = [numpy.float64, numpy.complex128] if double else [numpy.float32, numpy.complex64]

	def wrapper(ctx, shape, dtype, batch, fast_math):
		x, y, z = getDimensions(shape)
		if x * y * z * batch * dtype().nbytes > buffer_size * 1024 * 1024:
			return

		try:
			return testErrors(ctx, shape, dtype, batch, fast_math)
		except Exception, e:
			print "failed: " + str(ctx) + ", " + \
				str(shape) + ", batch " + str(batch) + \
				", dtype " + str(dtype) + ": " + str(e)
			raise

	errors = []

	for cuda in [True, False]:
		if cuda and not test_cuda:
			continue
		if not cuda and not test_opencl:
			continue

		ctx = createContext(cuda)
		for dtype in dtypes:
			assert dtype not in [numpy.float64, numpy.complex128] or \
				ctx.supportsDouble(), "Default device does not support double precision"

			for batch in batch_sizes:
				for shape in shapes:
					errors.append(wrapper(ctx, shape, dtype,
						batch, fast_math))

	inplace_errors = numpy.array([x[0] for x in errors if x is not None])
	numpy_diffs = numpy.array([x[1] for x in errors if x is not None])

	for arr, name in ((inplace_errors, "pyfft errors"),
			(numpy_diffs, "pyfft-numpy diffs")):

		print "* " + name + ":"
		print "min: " + str(numpy.min(arr))
		print "max: " + str(numpy.max(arr))
		print "avg: " + str(numpy.sum(arr) / arr.size)


if __name__ == "__main__":
	run(isCudaAvailable(), isCLAvailable(), DEFAULT_BUFFER_SIZE, True, False)
	run(isCudaAvailable(), isCLAvailable(), DEFAULT_BUFFER_SIZE, False, False)
