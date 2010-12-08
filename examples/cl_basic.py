from pyfft.cl import Plan
import numpy
import pyopencl as cl
import pyopencl.array as cl_array

# initialize context
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# create plan
plan = Plan((16, 16), queue=queue)

# prepare data
data = numpy.ones((16, 16), dtype=numpy.complex64)
gpu_data = cl_array.to_device(ctx, queue, data)
print gpu_data

# forward transform
plan.execute(gpu_data.data)
result = gpu_data.get()
print result

# inverse transform
plan.execute(gpu_data.data, inverse=True)
result = gpu_data.get()
error = numpy.abs(numpy.sum(numpy.abs(data) - numpy.abs(result)) / data.size)
print error < 1e-6
