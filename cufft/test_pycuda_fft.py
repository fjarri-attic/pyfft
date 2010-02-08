import numpy.fft as cpu

import pycuda.autoinit
import pycuda.gpuarray as gpuarray

from pycuda_fft import *
from test_init import *

##########
# 1D FFT #
##########
# n = 4
n = 1024 * 16
data_shape = (n,)
h_x = randn(*data_shape).astype('complex64')
h_x.imag = randn(*data_shape)
h_y = randn(*data_shape).astype('complex64')
h_y.imag = randn(*data_shape)

d_x = gpuarray.to_gpu(h_x)
d_y = gpuarray.to_gpu(h_y)

gpu_fft(d_x, d_y)
print d_x
gpu_result = d_y.get()
cpu_result = cpu.fft(h_x)
print 'fft: relative error =', relative_error(gpu_result, cpu_result)

gpu_fft(d_x)
gpu_result = d_x.get()
cpu_result = cpu.fft(h_x)
print 'in-place fft: relative error =', relative_error(gpu_result, cpu_result)
d_x = gpuarray.to_gpu(h_x)

gpu_ifft(d_x, d_y)
gpu_result = d_y.get()/d_x.size
cpu_result = cpu.ifft(h_x)
print 'ifft: relative error =', relative_error(gpu_result, cpu_result)

gpu_ifft(d_x)
gpu_result = d_x.get()/d_x.size
cpu_result = cpu.ifft(h_x)
print 'in-place ifft: relative error =', relative_error(gpu_result, cpu_result)
d_x = gpuarray.to_gpu(h_x)

##########
# 2D FFT #
##########
# n = 16
# n = 1024* 16
n = 256
data_shape = (n,n)
h_x = randn(*data_shape).astype('complex64')
h_x.imag = randn(*data_shape)
h_y = randn(*data_shape).astype('complex64')
h_y.imag = randn(*data_shape)

d_x = gpuarray.to_gpu(h_x)
d_y = gpuarray.to_gpu(h_y)

gpu_fft2(d_x, d_y)
gpu_result = d_y.get()
cpu_result = cpu.fft2(h_x)
print 'fft2: relative error =', relative_error(gpu_result, cpu_result)

gpu_fft2(d_x)
gpu_result = d_x.get()
cpu_result = cpu.fft2(h_x)
print 'in-place fft2: relative error =', relative_error(gpu_result, cpu_result)
d_x = gpuarray.to_gpu(h_x)

gpu_ifft2(d_x, d_y)
gpu_result = d_y.get()/d_x.size
cpu_result = cpu.ifft2(h_x)
print 'ifft2: relative error =', relative_error(gpu_result, cpu_result)

gpu_ifft2(d_x)
gpu_result = d_x.get()/d_x.size
cpu_result = cpu.ifft2(h_x)
print 'in-place ifft2: relative error =', relative_error(gpu_result, cpu_result)
d_x = gpuarray.to_gpu(h_x)

###############
# 1D FFT (R2C)#
###############
n = 256
input_shape = (n,)
output_shape = (n/2+1,)

h_x = randn(*input_shape).astype('float32')
d_x = gpuarray.to_gpu(h_x)
d_y = pycuda.gpuarray.GPUArray(output_shape, 'complex64')

cpu_result = cpu.rfft(h_x)
gpu_rfft(d_x,d_y)
gpu_result = d_y.get()
print 'rfft: relative error =', relative_error(gpu_result, cpu_result)

###############
# 1D FFT (C2R)#
###############
n = 256
input_shape = (n/2+1,)
output_shape = (n,)

h_x = randn(*input_shape).astype('complex64')
h_x.imag = randn(*input_shape).astype('complex64')
d_x = gpuarray.to_gpu(h_x)
d_y = pycuda.gpuarray.GPUArray(output_shape, 'float32')

cpu_result = cpu.irfft(h_x)
gpu_irfft(d_x,d_y)
gpu_result = d_y.get()/d_y.size
print 'irfft: relative error =', relative_error(gpu_result, cpu_result)

###############
# 2D FFT (R2C)#
###############
n = 256
input_shape = (n,n)
output_shape = (n,n/2+1)

h_x = randn(*input_shape).astype('float32')
d_x = gpuarray.to_gpu(h_x)
d_y = pycuda.gpuarray.GPUArray(output_shape, 'complex64')

cpu_result = cpu.rfft2(h_x)
gpu_rfft2(d_x,d_y)
gpu_result = d_y.get()
print 'rfft2: relative error =', relative_error(gpu_result, cpu_result)

###############
# 2D FFT (R2C)#
###############
n = 256
input_shape = (n,n/2+1)
output_shape = (n,n)

h_x = randn(*input_shape).astype('complex64')
h_x.imag = randn(*input_shape).astype('complex64')

d_x = gpuarray.to_gpu(h_x)
d_y = pycuda.gpuarray.GPUArray(output_shape, 'float32')

cpu_result = cpu.irfft2(h_x)
gpu_irfft2(d_x,d_y)
gpu_result = d_y.get()/d_y.size
print 'rfft2: relative error =', relative_error(gpu_result, cpu_result)
