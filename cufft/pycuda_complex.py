"""This code is written by Ying Wai (Daniel) Fan, Emory University, Dec 2009.
   Add complex number (single precision) support to some of PyCUDA funcitons."""
import numpy
from pytools import memoize
import pycuda
from pycuda.elementwise import get_elwise_kernel

def _all_complex(dtype_x, dtype_y, dtype_z):
   """Check if all three datatypes are complex64."""
   complex_arg_count = [dtype_x, dtype_y, dtype_z].count(numpy.complex64)
   if complex_arg_count == 0:
       return False
   elif complex_arg_count == 3:
       return True
   else:
       raise TypeError('Mixed operations between real and complex types not implemented.')

##########
# axpbyz #
##########
get_axpbyz_kernel_original = pycuda.elementwise.get_axpbyz_kernel

@memoize
def get_axpbyz_kernel(dtype_x, dtype_y, dtype_z):
    if _all_complex(dtype_x, dtype_y, dtype_z):
        return get_elwise_kernel(
            "float a, float2 *x, float b, float2 *y, float2 *z",
            "z[i].x = a*x[i].x + b*y[i].x; z[i].y = a*x[i].y + b*y[i].y",
            "axpbyz")
    else:
        return get_axpbyz_kernel_original(dtype_x, dtype_y, dtype_z) 

pycuda.elementwise.get_axpbyz_kernel = get_axpbyz_kernel

############
# multiply #
############
get_multiply_kernel_original = pycuda.elementwise.get_multiply_kernel

@memoize
def get_multiply_kernel(dtype_x, dtype_y, dtype_z):
    if _all_complex(dtype_x, dtype_y, dtype_z):
        return get_elwise_kernel(
            "float2 *x, float2 *y, float2 *z",
            "z[i].x = x[i].x * y[i].x - x[i].y * y[i].y; \
             z[i].y = x[i].x * y[i].y + x[i].y * y[i].x",
            "multiply")
    else:
        return get_multiply_kernel_original(dtype_x, dtype_y, dtype_z) 

pycuda.elementwise.get_multiply_kernel = get_multiply_kernel

def multiply(x, y, z, add_timer=None, stream=None):
    """Compute ``z = x * y``."""
    assert x.shape == y.shape
    
    func = get_multiply_kernel(x.dtype, y.dtype, z.dtype)
    func.set_block_shape(*x._block)
    
    if add_timer is not None:
        add_timer(3*x.size, func.prepared_timed_call(x._grid, 
                  x.gpudata, y.gpudata, z.gpudata, x.mem_size))
    else:
        func.prepared_async_call(x._grid, stream,
                  x.gpudata, y.gpudata, z.gpudata, x.mem_size)
        
##########
# divide #
##########
get_divide_kernel_original = pycuda.elementwise.get_divide_kernel

@memoize
def get_divide_kernel(dtype_x, dtype_y, dtype_z):
    if _all_complex(dtype_x, dtype_y, dtype_z):
        return get_elwise_kernel(
            "float2 *x, float2 *y, float2 *z",
            "norm_squared  = y[i].x * y[i].x + y[i].y * y[i].y; \
             z[i].x = (x[i].x * y[i].x + x[i].y * y[i].y) / norm_squared; \
             z[i].y = (-x[i].x * y[i].y + x[i].y * y[i].x) / norm_squared",
            "divide", beforeloop="float norm_squared;")
    else:
        return get_get_divide_kernel_original(dtype_x, dtype_y, dtype_z) 

pycuda.elementwise.get_divide_kernel = get_divide_kernel

######################
# conjugate_multiply #
######################
@memoize
def get_conjugate_multiply_kernel(dtype_x, dtype_y, dtype_z):
     return get_elwise_kernel(
         "float2 *x, float2 *y, float2 *z",
         "z[i].x = x[i].x * y[i].x + x[i].y * y[i].y; \
          z[i].y = x[i].y * y[i].x - x[i].x * y[i].y",
         "conjugate_multiply")

def conjugate_multiply(x, y, z, add_timer=None, stream=None):
    """Compute ``z = conj(x) * y``."""
    assert x.shape == y.shape
    
    func = get_conjugate_multiply_kernel(x.dtype, y.dtype, z.dtype)
    func.set_block_shape(*x._block)
    
    if add_timer is not None:
        add_timer(3*x.size, func.prepared_timed_call(x._grid, 
                  x.gpudata, y.gpudata, z.gpudata, x.mem_size))
    else:
        func.prepared_async_call(x._grid, stream,
                  x.gpudata, y.gpudata, z.gpudata, x.mem_size)
        
####################
# divide_by_scalar #
####################
@memoize
def get_divide_by_scalar_kernel():
     return get_elwise_kernel(
         "float2 *x, float y, float2 *z",
         "z[i].x = x[i].x / y; z[i].y = x[i].y / y",
         "divide_by_scalar")

def divide_by_scalar(x, y, z, add_timer=None, stream=None):
    """Compute ``z = x / y``, where y is a real scalar."""
    assert x.shape == z.shape
    
    func = get_divide_by_scalar_kernel()
    func.set_block_shape(*x._block)
    
    if add_timer is not None:
        add_timer(3*x.size, func.prepared_timed_call(x._grid, 
                  x.gpudata, y, z.gpudata, x.mem_size))
    else:
        func.prepared_async_call(x._grid, stream,
                  x.gpudata, y, z.gpudata, x.mem_size)
