import numpy
import time

COMPLEX_DTYPES = [numpy.complex64, numpy.complex128]
DOUBLE_DTYPES = [numpy.float64, numpy.complex128]

DEFAULT_BUFFER_SIZE = 16


def isCLAvailable():
	res = False
	try:
		import pyopencl
		res = True
	except:
		pass
	return res

def isCudaAvailable():
	res = False
	try:
		import pycuda.driver
		res = True
	except:
		pass
	return res


class CudaContext:

	def __init__(self):
		import pycuda.tools
		import pycuda.driver
		pycuda.driver.init()
		self.context = pycuda.tools.make_default_context()

	def __del__(self):
		self.context.pop()

	def allocate(self, shape, dtype):
		import pycuda.gpuarray as gpuarray
		return gpuarray.GPUArray(shape, dtype=dtype)

	def toGpu(self, data):
		import pycuda.gpuarray as gpuarray
		return gpuarray.to_gpu(data)

	def fromGpu(self, gpu_buf, target_shape, target_dtype):
		return gpu_buf.get()

	def getMemoryPool(self):
		import pycuda.tools
		return pycuda.tools.DeviceMemoryPool()

	def getPlan(self, *args, **kwds):
		import pyfft.cuda
		return pyfft.cuda.Plan(*args, **kwds)

	def startTimer(self):
		import pycuda.driver
		self._start = pycuda.driver.Event().record()
		self._stop = pycuda.driver.Event()

	def stopTimer(self):
		self._stop.record()
		self._stop.synchronize()
		return self._stop.time_since(self._start) / 1000.0

	def supportsDouble(self):
		major, minor = self.context.get_device().compute_capability()
		return (major == 1 and minor == 3) or major >= 2

	def __str__(self):
		return "cuda"


class CLContext:

	def __init__(self):
		import pyopencl as cl

		# Choose first GPU device. Not using commented line below,
		# because we need context with only one device (to avoid
		# complications)
		for p in cl.get_platforms():
			for d in p.get_devices(device_type=cl.device_type.GPU):
				self.context = cl.Context(devices=[d])
				return

		#self.context = cl.Context(dev_type=cl.device_type.GPU)

	def _createQueue(self):
		import pyopencl as cl
		return cl.CommandQueue(self.context)

	def allocate(self, shape, dtype):
		import pyopencl as cl
		x, y, z = getDimensions(shape)
		return cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=(x * y * z * dtype.itemsize))

	def toGpu(self, data):
		import pyopencl as cl
		gpu_buf = self.allocate(data.shape, data.dtype)
		queue = self._createQueue()
		cl.enqueue_write_buffer(queue, gpu_buf, data).wait()
		return gpu_buf

	def fromGpu(self, gpu_buf, target_shape, target_dtype):
		import pyopencl as cl
		data = numpy.empty(target_shape, target_dtype)
		queue = self._createQueue()
		cl.enqueue_read_buffer(queue, gpu_buf, data).wait()
		return data

	def getPlan(self, *args, **kwds):
		import pyfft.cl
		return pyfft.cl.Plan(*args, **kwds)

	def startTimer(self):
		self._start = time.time()

	def stopTimer(self):
		self._stop = time.time()
		return self._stop - self._start

	def supportsDouble(self):
		return "cl_khr_fp64" in self.context.devices[0].extensions

	def __str__(self):
		return "cl"


def createContext(cuda):
	if cuda:
		return CudaContext()
	else:
		return CLContext()


def log2(n):
	pos = 0
	for pow in [16, 8, 4, 2, 1]:
		if n >= 2 ** pow:
			n /= (2 ** pow)
			pos += pow
	return pos

def getDimensions(shape):
	if isinstance(shape, int):
		return shape, 1, 1
	elif isinstance(shape, tuple):
		if len(shape) == 1:
			return shape[0], 1, 1
		elif len(shape) == 2:
			return shape[0], shape[1], 1
		elif len(shape) == 3:
			return shape[0], shape[1], shape[2]

def difference(arr1, arr2, batch):
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

def getTestArray(shape, dtype, batch):
	arrays = []
	for i in xrange(batch):
		#arrays.append(numpy.ones(shape, dtype=dtype))
		arrays.append(numpy.random.randn(*shape).astype(dtype))

	return numpy.concatenate(arrays)

def getTestData(shape, dtype, batch):
	if dtype in COMPLEX_DTYPES:
		if dtype == numpy.complex64:
			float_dtype = numpy.float32
		else:
			float_dtype = numpy.float64

		return (getTestArray(shape, float_dtype, batch) + \
			1j * getTestArray(shape, float_dtype, batch)).astype(dtype)
	else:
		return getTestArray(shape, dtype, batch), \
			getTestArray(shape, dtype, batch)
