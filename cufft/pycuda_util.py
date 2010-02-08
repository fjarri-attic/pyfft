from ctypes import cast, c_float, POINTER, Structure
from numpy import complex64
from pycuda.gpuarray import GPUArray, to_gpu

class float2(Structure):
    pass
float2._fields_ = [
    ('x', c_float),
    ('y', c_float),
]

def pointer(gpuarray):
  """Return the pointer to the linear memory held by a GPUArray object."""
  if gpuarray.dtype != complex64:
      return cast(int(gpuarray.gpudata), POINTER(c_float))
  else:
      return cast(int(gpuarray.gpudata), POINTER(float2))

def to_cpu(x):
    if isinstance(x, GPUArray):
        return x.get()
    else:
        return x

def to_gpu(x):
    if isinstance(x, GPUArray):
        return x
    else:
        return to_gpu(x)

def sub2ind(shape, I, J, row_major=True):
    if row_major:
        ind = (I % shape[0]) * shape[1] + (J % shape[1])
    else:
        ind = (J % shape[1]) * shape[0] + (I % shape[0])
    return ind
