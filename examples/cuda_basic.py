from pyfft.cuda import Plan
import numpy
from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

# initialize context
cuda.init()
context = make_default_context()
stream = cuda.Stream()

# create plan
plan = Plan((16, 16), stream=stream)

# prepare data
data = numpy.ones((16, 16), dtype=numpy.complex64)
gpu_data = gpuarray.to_gpu(data)
print gpu_data

# forward transform
plan.execute(gpu_data)
result = gpu_data.get()
print result

# inverse transform
plan.execute(gpu_data, inverse=True)
result = gpu_data.get()
error = numpy.abs(numpy.sum(numpy.abs(data) - numpy.abs(result)) / data.size)
print error < 1e-6

context.pop()
