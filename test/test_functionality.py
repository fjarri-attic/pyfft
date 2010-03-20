import unittest
import numpy

from helpers import *


class TestPlan(unittest.TestCase):

	def tearDown(self):
		del self.context

	def testShapes(self):
		for shape in [16, (16,), (16, 16), (16, 16, 16)]:
			plan = self.context.getPlan(shape, dtype=numpy.float32)

	def testTypes(self):
		dtypes = [numpy.float32, numpy.complex64]
		if self.context.supportsDouble():
			dtypes.extend([numpy.float64, numpy.complex128])

		for dtype in dtypes:
			plan = self.context.getPlan((16, 16), dtype=dtype)

	def testExecuteSignatureSplit(self):
		dtype = numpy.float32
		plan = self.context.getPlan((16,), dtype=dtype)
		a_gpu = self.context.toGpu(numpy.ones(16, dtype=dtype))
		b_gpu = self.context.toGpu(numpy.ones(16, dtype=dtype))
		c_gpu = self.context.toGpu(numpy.ones(16, dtype=dtype))
		d_gpu = self.context.toGpu(numpy.ones(16, dtype=dtype))

		# should work
		plan.execute(a_gpu, b_gpu)
		plan.execute(a_gpu, b_gpu, c_gpu, d_gpu)
		plan.execute(a_gpu, b_gpu, a_gpu, b_gpu)

	def testExecuteSignatureInterleaved(self):
		dtype = numpy.complex64
		plan = self.context.getPlan((16,), dtype=dtype)
		a_gpu = self.context.toGpu(numpy.ones(16, dtype=dtype))
		b_gpu = self.context.toGpu(numpy.ones(16, dtype=dtype))

		# should work
		plan.execute(a_gpu)
		plan.execute(a_gpu, b_gpu)
		plan.execute(a_gpu, a_gpu)

		# should not work
		self.assertRaises(TypeError, plan.execute, a_gpu, b_gpu, a_gpu, b_gpu, inverse=True)

	def testNormalize(self):
		dtype = numpy.complex64
		data = numpy.ones(16, dtype=dtype)

		for normalize in [True, False]:
			plan = self.context.getPlan((16,), normalize=normalize)
			a_gpu = self.context.toGpu(data)
			plan.execute(a_gpu)
			plan.execute(a_gpu, inverse=True)
			res = self.context.fromGpu(a_gpu, data.shape, data.dtype)

			coeff = 1 if normalize else 16

			error = numpy.sum(data * coeff - res) / 16
			self.assert_(error < 1e-6)

	def testAllocation(self):
		plan = self.context.getPlan((32, 32, 32), dtype=numpy.complex64)
		a_gpu = self.context.toGpu(numpy.ones((32, 32, 32), dtype=numpy.complex64))
		plan.execute(a_gpu)

	def testPrecreatedContext(self):
		plan = self.context.getPlan((16,), dtype=numpy.complex64,
			context=self.context.context)
		a_gpu = self.context.toGpu(numpy.ones((16,), dtype=numpy.complex64))
		plan.execute(a_gpu)

	def testWrongDataSize(self):
		self.assertRaises(ValueError, self.context.getPlan, (17,), dtype=numpy.complex64)

	def testWrongDataType(self):
		self.assertRaises(ValueError, self.context.getPlan, (16,), dtype=numpy.int32)

	def testWrongShape(self):
		self.assertRaises(ValueError, self.context.getPlan, (16, 16, 16, 16),
			dtype=numpy.complex64)
		self.assertRaises(ValueError, self.context.getPlan, "16",
			dtype=numpy.complex64)


class CudaPlan(TestPlan):

	def setUp(self):
		self.context = createContext(True)

	def testMempool(self):
		import pycuda.tools
		plan = self.context.getPlan((32, 32, 32), dtype=numpy.complex64,
			mempool=pycuda.tools.DeviceMemoryPool())

	def testExternalStream(self):
		import pycuda.driver
		stream = pycuda.driver.Stream()
		plan = self.context.getPlan((32, 32, 32), dtype=numpy.complex64,
			stream=stream)
		a_gpu = self.context.toGpu(numpy.ones((32, 32, 32), dtype=numpy.complex64))
		plan.execute(a_gpu)
		stream.synchronize()

	def testGetStream(self):
		plan = self.context.getPlan((32, 32, 32), dtype=numpy.complex64)
		a_gpu = self.context.toGpu(numpy.ones((32, 32, 32), dtype=numpy.complex64))
		stream = plan.execute(a_gpu, wait_for_finish=False)
		stream.synchronize()


class CLPlan(TestPlan):

	def setUp(self):
		self.context = createContext(False)

	def testExternalQueue(self):
		import pyopencl as cl
		queue = cl.CommandQueue(self.context.context)
		plan = self.context.getPlan((16,), dtype=numpy.complex64, queue=queue)

	def testGetQueue(self):
		plan = self.context.getPlan((32, 32, 32), dtype=numpy.complex64)
		a_gpu = self.context.toGpu(numpy.ones((32, 32, 32), dtype=numpy.complex64))
		queue = plan.execute(a_gpu, wait_for_finish=False)
		queue.finish()


def run(test_cuda, test_opencl):
	print "Running functionality tests..."

	suites = []
	if test_cuda:
		suites.append(unittest.TestLoader().loadTestsFromTestCase(CudaPlan))

	if test_opencl:
		suites.append(unittest.TestLoader().loadTestsFromTestCase(CLPlan))

	all = unittest.TestSuite(suites)
	unittest.TextTestRunner(verbosity=1).run(all)

if __name__ == "__main__":
	run(isCudaAvailable(), isCLAvailable())
