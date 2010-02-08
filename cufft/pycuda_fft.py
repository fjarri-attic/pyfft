"""This code is written by Ying Wai (Daniel) Fan, Emory University, Dec 2009.
Part of this modules follows the design of Derek Anderson's pycublas.py and Nicolas Pinto's cufft.py"""
import ctypes
import transpose
from typenames import *
from ctypes import c_uint, c_int, c_float, POINTER

from pycuda_util import float2, pointer
import pycuda.gpuarray as gpuarray

#####################
# Wrappers to CUFFT #
#####################

import platform
if platform.system()=='Microsoft': libcufft = ctypes.windll.LoadLibrary('cufft.dll')
elif platform.system()=='Darwin':    libcufft = ctypes.cdll.LoadLibrary('/usr/local/cuda/lib/libcufft.dylib')
elif platform.system()=='Linux':     libcufft = ctypes.cdll.LoadLibrary('libcufft.so')
else:                              libcufft = ctypes.cdll.LoadLibrary('libcufft.so')

cufftReal = c_float
cufftComplex = float2
cufftResult = c_int
cufftHandle = c_uint
cufftType = c_int

cufftPlan1d = libcufft.cufftPlan1d
cufftPlan1d.restype = cufftResult
cufftPlan1d.argtypes = [POINTER(cufftHandle), c_int, cufftType, c_int]
cufftPlan1d.__doc__ = """cufftResult cufftPlan1d(cufftHandle * plan, int nx, cufftType type, int batch)"""

cufftPlan2d = libcufft.cufftPlan2d
cufftPlan2d.restype = cufftResult
cufftPlan2d.argtypes = [POINTER(cufftHandle), c_int, c_int, cufftType]
cufftPlan2d.__doc__ = """cufftResult cufftPlan2d(cufftHandle * plan, int nx, int ny, cufftType type)"""

cufftPlan3d = libcufft.cufftPlan3d
cufftPlan3d.restype = cufftResult
cufftPlan3d.argtypes = [POINTER(cufftHandle), c_int, c_int, c_int, cufftType]
cufftPlan3d.__doc__ = """cufftResult cufftPlan3d(cufftHandle * plan, int nx, int ny, int nz, cufftType type)"""

cufftDestroy = libcufft.cufftDestroy
cufftDestroy.restype = cufftResult
cufftDestroy.argtypes = [cufftHandle]
cufftDestroy.__doc__ = """cufftResult cufftDestroy(cufftHandle plan)"""

cufftExecC2C = libcufft.cufftExecC2C
cufftExecC2C.restype = cufftResult
cufftExecC2C.argtypes = [cufftHandle, POINTER(cufftComplex), POINTER(cufftComplex), c_int]
cufftExecC2C.__doc__ = """cufftResult cufftExecC2C(cufftHandle plan, cufftComplex * idata, cufftComplex * odata, int direction)"""

cufftExecR2C = libcufft.cufftExecR2C
cufftExecR2C.restype = cufftResult
cufftExecR2C.argtypes = [cufftHandle, POINTER(cufftReal), POINTER(cufftComplex)]
cufftExecR2C.__doc__ = """cufftResult cufftExecR2C(cufftHandle plan, cufftReal * idata, cufftComplex * odata)"""

cufftExecC2R = libcufft.cufftExecC2R
cufftExecC2R.restype = cufftResult
cufftExecC2R.argtypes = [cufftHandle, POINTER(cufftComplex), POINTER(cufftReal)]
cufftExecC2R.__doc__ = """cufftResult cufftExecC2R(cufftHandle plan, cufftComplex * idata, cufftReal * odata)"""

CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

CUFFT_R2C = 0x2a     # Real to Complex (interleaved)
CUFFT_C2R = 0x2c     # Complex (interleaved) to Real
CUFFT_C2C = 0x29     # Complex to Complex, interleaved
CUFFT_D2Z = 0x6a     # Double to Double-Complex
CUFFT_Z2D = 0x6c     # Double-Complex to Double
CUFFT_Z2Z = 0x69     # Double-Complex to Double-Complex

####################
# Useful utilities #
####################
CUFFT_SUCCESS        = 0x0
CUFFT_INVALID_PLAN   = 0x1
CUFFT_ALLOC_FAILED   = 0x2
CUFFT_INVALID_TYPE   = 0x3
CUFFT_INVALID_VALUE  = 0x4
CUFFT_INTERNAL_ERROR = 0x5
CUFFT_EXEC_FAILED    = 0x6
CUFFT_SETUP_FAILED   = 0x7
CUFFT_INVALID_SIZE   = 0x8
CUFFT_RESULT = {
  CUFFT_SUCCESS        : 'CUFFT_SUCCESS',
  CUFFT_INVALID_PLAN   : 'CUFFT_INVALID_PLAN',
  CUFFT_ALLOC_FAILED   : 'CUFFT_ALLOC_FAILED',
  CUFFT_INVALID_TYPE   : 'CUFFT_INVALID_TYPE',
  CUFFT_INVALID_VALUE  : 'CUFFT_INVALID_VALUE',
  CUFFT_INTERNAL_ERROR : 'CUFFT_INTERNAL_ERROR',
  CUFFT_EXEC_FAILED    : 'CUFFT_EXEC_FAILED',
  CUFFT_SETUP_FAILED   : 'CUFFT_SETUP_FAILED',
  CUFFT_INVALID_SIZE   : 'CUFFT_INVALID_SIZE'
}

def check_cufft_result(result):
  if result != CUFFT_SUCCESS:
    raise Exception(CUFFT_RESULT[result])

class fftplan(cufftHandle):
    def __del__(self):
        cufftDestroy(self)

##############################
# 1D FFT Complex <-> Complex #
##############################
def get_1dplan(shape, batch=1):
    plan = fftplan()
    cufftPlan1d(plan, shape[0], CUFFT_C2C, batch)
    return plan

def _fft(direction, x, y=None, plan=None):
    if plan is None:
        plan = get_1dplan(x.shape)

    idata = pointer(x)

    if y is None:
        odata = idata
    else:
        odata = pointer(y)

    result = cufftExecC2C(plan, idata, odata, direction)
    check_cufft_result(result)

def gpu_fft(x, y=None, plan=None):
    _fft(CUFFT_FORWARD, x, y, plan)

def gpu_ifft(x, y=None, plan=None):
    _fft(CUFFT_INVERSE, x, y, plan)

class CUFFTPlan:

	def __init__(self, x, y, z, batch):
		if z == 1:
			if y == 1:
				self.dim = 1
			else:
				self.dim = 2
		else:
			self.dim = 3

		self.x = x
		self.y = y
		self.z = z

		self.batch = batch
		self.temp = gpuarray.GPUArray((x * y * z * batch,), dtype=numpy.complex64)
		self.xplan = get_1dplan((self.x,), batch=self.y * self.z * self.batch)

		if self.dim == 2:
			self.yplan = get_1dplan((self.y,), batch=self.x * self.z * self.batch)

		if self.dim == 3:
			self.zplan = get_1dplan((self.z,), batch=self.x * self.y * self.batch)

		self.tr = transpose.Transpose(complex32)

	def _execute1d(self, idata, odata, direction):
		_fft(direction, idata, odata, plan=self.xplan)

	def _execute2d(self, idata, odata, direction):
		self._execute1d(idata, odata, direction)
		self.tr(self.temp.gpudata, odata.gpudata, self.x, self.y, self.batch * self.z)
		_fft(direction, self.temp, plan=self.yplan)
		self.tr(odata.gpudata, self.temp.gpudata, self.y, self.x, self.batch * self.z)

	def _execute3d(self, idata, odata, direction):
		self._execute2d(idata, odata, direction)
		self.tr(self.temp.gpudata, odata.gpudata, self.x * self.y, self.z, self.batch)
		_fft(direction, self.temp, plan=self.zplan)
		self.tr(odata.gpudata, self.temp.gpudata, self.z, self.x * self.y, self.batch)

	def execute(self, idata, odata, direction):
		if self.dim == 1:
			self._execute1d(idata, odata, direction)
		elif self.dim == 2:
			self._execute2d(idata, odata, direction)
		else:
			self._execute3d(idata, odata, direction)

##############################
# 2D FFT Complex <-> Complex #
##############################
def get_2dplan(shape):
    plan = fftplan()
    cufftPlan2d(plan, shape[1], shape[0], CUFFT_C2C)
    return plan

def _fft2(direction, x, y=None, plan=None):
    if plan is None:
        plan = get_2dplan(x.shape)

    idata = pointer(x)

    if y is None:
        odata = idata
    else:
        odata = pointer(y)

    result = cufftExecC2C(plan, idata, odata, direction)
    check_cufft_result(result)

def gpu_fft2(x, y=None, plan=None):
    _fft2(CUFFT_FORWARD, x, y, plan)

def gpu_ifft2(x, y=None, plan=None):
    _fft2(CUFFT_INVERSE, x, y, plan)

###########################
# 1D FFT Real <-> Complex #
###########################
def get_1dplan_r2c(shape,batch=1):
    plan = fftplan()
    cufftPlan1d(plan, shape[0], CUFFT_R2C, batch)
    return plan

def get_1dplan_c2r(shape,batch=1):
    plan = fftplan()
    cufftPlan1d(plan, shape[0], CUFFT_C2R, batch)
    return plan

def gpu_rfft(x, y, plan=None):
    idata = pointer(x)
    odata = pointer(y)
    if plan is None:
        plan = get_1dplan_r2c((x.size,))
    result = cufftExecR2C(plan, idata, odata)
    check_cufft_result(result)

def gpu_irfft(x, y, plan=None):
    idata = pointer(x)
    odata = pointer(y)
    if plan is None:
        plan = get_1dplan_c2r((y.size,))
    result = cufftExecC2R(plan, idata, odata)
    check_cufft_result(result)

###########################
# 2D FFT Real <-> Complex #
###########################
def get_2dplan_r2c(shape):
    plan = fftplan()
    cufftPlan2d(plan, shape[0], shape[1], CUFFT_R2C)
    return plan

def get_2dplan_c2r(shape):
    plan = fftplan()
    cufftPlan2d(plan, shape[0], shape[1], CUFFT_C2R)
    return plan

def gpu_rfft2(x, y, plan=None):
    idata = pointer(x)
    odata = pointer(y)
    if plan is None:
        plan = get_2dplan_r2c(x.shape)
    result = cufftExecR2C(plan, idata, odata)
    check_cufft_result(result)

def gpu_irfft2(x, y, plan=None):
    idata = pointer(x)
    odata = pointer(y)
    if plan is None:
        plan = get_2dplan_c2r(y.shape)
    result = cufftExecC2R(plan, idata, odata)
    check_cufft_result(result)

def malloc_for_r2c(idata):
    from pycuda.gpuarray import GPUArray
    input_shape = idata.shape
    output_shape = (input_shape[0], input_shape[1]/2 + 1)
    temp = GPUArray(output_shape, 'complex64')
    return temp
