import unittest
import numpy

from helpers import *


class TestPlan(unittest.TestCase):

	def tearDown(self):
		del self.context

	def testShapes(self):
		for shape in [16, (16,), (16, 16), (16, 16, 16)]:
			plan = self.context.getPlan(shape, dtype=self.scalar,
				context=self.context.context)

	def testTypes(self):
		dtypes = [self.scalar, self.complex]

		for dtype in dtypes:
			plan = self.context.getPlan((16, 16), dtype=dtype,
				context=self.context.context)

	def testExecuteSignatureSplit(self):
		dtype = self.scalar
		plan = self.context.getPlan((16,), dtype=dtype,
			context=self.context.context)
		a_gpu = self.context.toGpu(numpy.ones(16, dtype=dtype))
		b_gpu = self.context.toGpu(numpy.ones(16, dtype=dtype))
		c_gpu = self.context.toGpu(numpy.ones(16, dtype=dtype))
		d_gpu = self.context.toGpu(numpy.ones(16, dtype=dtype))

		# should work
		plan.execute(a_gpu, b_gpu)
		plan.execute(a_gpu, b_gpu, c_gpu, d_gpu)
		plan.execute(a_gpu, b_gpu, a_gpu, b_gpu)

	def testExecuteSignatureInterleaved(self):
		dtype = self.complex
		plan = self.context.getPlan((16,), dtype=dtype,
			context=self.context.context)
		a_gpu = self.context.toGpu(numpy.ones(16, dtype=dtype))
		b_gpu = self.context.toGpu(numpy.ones(16, dtype=dtype))

		# should work
		plan.execute(a_gpu)
		plan.execute(a_gpu, b_gpu)
		plan.execute(a_gpu, a_gpu)

		# should not work
		self.assertRaises(TypeError, plan.execute, a_gpu, b_gpu, a_gpu, b_gpu, inverse=True)

	def testNormalize(self):
		dtype = self.complex
		data = numpy.ones(16, dtype=dtype)

		for normalize in [True, False]:
			plan = self.context.getPlan(data.shape, normalize=normalize,
				dtype=dtype, context=self.context.context)
			a_gpu = self.context.toGpu(data)

			# Test forward transform
			# Should be the same as numpy regardless of the 'normalize' value
			plan.execute(a_gpu)
			numpy_res = numpy.fft.fft(data)
			res = self.context.fromGpu(a_gpu, data.shape, data.dtype)
			error = numpy.sum(numpy.abs(numpy_res - res)) / data.size
			self.assert_(error < 1e-6)

			# Test backward transform
			plan.execute(a_gpu, inverse=True)
			res = self.context.fromGpu(a_gpu, data.shape, data.dtype)

			coeff = 1 if normalize else data.size

			error = numpy.sum(numpy.abs(data * coeff - res)) / data.size
			self.assert_(error < 1e-6)

	def testScale(self):
		dtype = self.complex
		data = numpy.ones(16, dtype=dtype)

		for scale in [1.0, 10.0]:
			plan = self.context.getPlan(data.shape, scale=scale,
				dtype=dtype, context=self.context.context, normalize=True)
			a_gpu = self.context.toGpu(data)

			# Forward - scaling must be applied
			# (result must be 'scale' times bigger than numpy result)
			plan.execute(a_gpu)
			numpy_res = numpy.fft.fft(data)
			res = self.context.fromGpu(a_gpu, data.shape, data.dtype)
			error = numpy.sum(numpy.abs(numpy_res * scale - res)) / data.size
			self.assert_(error < 1e-6)

			# Backward - inverse scaling must be applied, returning things to normal
			plan.execute(a_gpu, inverse=True)
			res = self.context.fromGpu(a_gpu, data.shape, data.dtype)
			error = numpy.sum(numpy.abs(data - res)) / data.size
			self.assert_(error < 1e-6)

	def testFastMath(self):
		dtype = self.complex
		data = numpy.ones(8192, dtype=dtype)

		for fast_math in [True, False]:
			plan = self.context.getPlan(data.shape, normalize=True,
				dtype=dtype, context=self.context.context, fast_math=fast_math)
			a_gpu = self.context.toGpu(data)
			plan.execute(a_gpu)
			plan.execute(a_gpu, inverse=True)
			res = self.context.fromGpu(a_gpu, data.shape, data.dtype)

			error = numpy.sum(numpy.abs(data - res)) / data.size
			self.assert_(error < 1e-6)

	def testAllocation(self):
		plan = self.context.getPlan((32, 32, 32), dtype=self.complex,
			context=self.context.context)
		a_gpu = self.context.toGpu(numpy.ones((32, 32, 32), dtype=self.complex))
		plan.execute(a_gpu)

	def testPrecreatedContext(self):
		plan = self.context.getPlan((16,), dtype=self.complex,
			context=self.context.context)
		a_gpu = self.context.toGpu(numpy.ones((16,), dtype=self.complex))
		plan.execute(a_gpu)

	def testWrongDataSize(self):
		self.assertRaises(ValueError, self.context.getPlan, (17,), dtype=self.complex)

	def testWrongDataType(self):
		self.assertRaises(ValueError, self.context.getPlan, (16,), dtype=numpy.int32)

	def testWrongShape(self):
		self.assertRaises(ValueError, self.context.getPlan, (16, 16, 16, 16),
			dtype=self.complex)
		self.assertRaises(ValueError, self.context.getPlan, "16",
			dtype=self.complex)


class CudaPlan(TestPlan):

	def setUp(self):
		self.context = createContext(True)

	def testMempool(self):
		import pycuda.tools
		plan = self.context.getPlan((32, 32, 32), dtype=self.complex,
			mempool=pycuda.tools.DeviceMemoryPool())

	def testExternalStream(self):
		import pycuda.driver
		stream = pycuda.driver.Stream()
		plan = self.context.getPlan((32, 32, 32), dtype=self.complex,
			stream=stream)
		a_gpu = self.context.toGpu(numpy.ones((32, 32, 32), dtype=self.complex))
		plan.execute(a_gpu)
		stream.synchronize()

	def testGetStream(self):
		plan = self.context.getPlan((32, 32, 32), dtype=self.complex)
		a_gpu = self.context.toGpu(numpy.ones((32, 32, 32), dtype=self.complex))
		stream = plan.execute(a_gpu, wait_for_finish=False)
		stream.synchronize()


class CLPlan(TestPlan):

	def setUp(self):
		self.context = createContext(False)

	def testExternalQueue(self):
		import pyopencl as cl
		queue = cl.CommandQueue(self.context.context)
		plan = self.context.getPlan((16,), dtype=self.complex, queue=queue)

	def testGetQueue(self):
		plan = self.context.getPlan((32, 32, 32), dtype=self.complex,
			context=self.context.context)
		a_gpu = self.context.toGpu(numpy.ones((32, 32, 32), dtype=self.complex))
		queue = plan.execute(a_gpu, wait_for_finish=False)
		queue.finish()

	def testNoContextNoQueue(self):
		self.assertRaises(ValueError, self.context.getPlan, (32, 32, 32), dtype=self.complex)


def setPrecision(cls, is_double):
	class Temp(cls):
		scalar = numpy.float64 if is_double else numpy.float32
		complex = numpy.complex128 if is_double else numpy.complex64

	Temp.__name__ = cls.__name__ + "_" + ("double" if is_double else "float")
	return Temp

def run(test_cuda, test_opencl):
	print "Running functionality tests..."

	suites = []
	add_suite = lambda cls, is_double: suites.append(
		unittest.TestLoader().loadTestsFromTestCase(setPrecision(cls, is_double)))

	if test_cuda:
		add_suite(CudaPlan, False)

		ctx = createContext(True)
		double_available = ctx.supportsDouble()
		del ctx

		if double_available:
			add_suite(CudaPlan, True)

	if test_opencl:
		add_suite(CLPlan, False)

		ctx = createContext(False)
		double_available = ctx.supportsDouble()
		del ctx

		if double_available:
			add_suite(CLPlan, True)

	all = unittest.TestSuite(suites)
	unittest.TextTestRunner(verbosity=1).run(all)

if __name__ == "__main__":
	run(isCudaAvailable(), isCLAvailable())
