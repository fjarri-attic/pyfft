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


class CudaPlan(TestPlan):

	def setUp(self):
		self.context = createContext(True)


class CLPlan(TestPlan):

	def setUp(self):
		self.context = createContext(False)

	def testPrecreatedQueue(self):
		import pyopencl as cl
		queue = cl.CommandQueue(self.context.context)
		plan = self.context.getPlan((16,), dtype=numpy.complex64, queue=queue)


def run(test_cuda, test_opencl):
	print "Running functionality tests..."

	suites = []
	if test_cuda:
		suites.append(unittest.TestLoader().loadTestsFromTestCase(CudaPlan))

	if test_opencl:
		suites.append(unittest.TestLoader().loadTestsFromTestCase(CLPlan))

	all = unittest.TestSuite(suites)
	unittest.TextTestRunner(verbosity=3).run(all)

if __name__ == "__main__":
	run(isCudaAvailable(), isCLAvailable())
