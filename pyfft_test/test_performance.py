from helpers import *

def testPerformance(ctx, shape, buffer_size):

	dtype = numpy.complex64
	buf_size_bytes = buffer_size * 1024 * 1024
	value_size = dtype().nbytes
	iterations = 10

	x, y, z = getDimensions(shape)
	batch = buf_size_bytes / (x * y * z * value_size)

	if batch == 0:
		print "Buffer size is too big, skipping test"
		return

	data = getTestData(shape, dtype, batch=batch)

	a_gpu = ctx.toGpu(data)
	b_gpu = ctx.allocate(data.shape, data.dtype)

	plan = ctx.getPlan(shape, context=ctx.context, wait_for_finish=True)

	gflop = 5.0e-9 * (log2(x) + log2(y) + log2(z)) * x * y * z * batch

	plan.execute(a_gpu, b_gpu, batch=batch) # warming up
	ctx.startTimer()
	for i in xrange(iterations):
		plan.execute(a_gpu, b_gpu, batch=batch)
	t_pyfft = ctx.stopTimer() / iterations

	print "* " + str(ctx) + ", " + str(shape) + ", batch " + str(batch) + ": " + \
		str(t_pyfft * 1000) + " ms, " + str(gflop / t_pyfft) + " GFLOPS"

def run(test_cuda, test_opencl, buffer_size):
	print "Running performance tests..."

	shapes = [
		(16,), (1024,), (8192,), # 1D
		(16, 16), (128, 128), (1024, 1024), # 2D
		(16, 16, 16), (32, 32, 128), (128, 128, 128) # 3D
	]

	for cuda in [True, False]:
		if cuda and not test_cuda:
			continue
		if not cuda and not test_opencl:
			continue

		ctx = createContext(cuda)
		for shape in shapes:
			testPerformance(ctx, shape, buffer_size)

		del ctx # just in case, to make sure it is deleted before the next one is created

if __name__ == "__main__":
	run(isCudaAvailable(), isCLAvailable(), DEFAULT_BUFFER_SIZE)
