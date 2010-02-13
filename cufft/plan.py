from pycuda_fft import CUFFT_FORWARD, CUFFT_INVERSE, _fft, get_1dplan
import transpose

import pycuda.gpuarray as gpuarray
import numpy

_FFT_1D = 1
_FFT_2D = 2
_FFT_3D = 3


class CUFFTPlan:

	def __init__(self, x, y=None, z=None, batch=1, dtype=numpy.complex64):
		if z is None:
			if y is None:
				self._dim = _FFT_1D
			else:
				self._dim = _FFT_2D
		else:
			self._dim = _FFT_3D

		self._x = x
		self._y = 1 if y is None else y
		self._z = 1 if z is None else z
		self._batch = batch

		if dtype != numpy.complex64:
			raise NotImplementedError("Only complex64 (single-precision) is currently supported")

		self._dtype = dtype

		typenames = {
			numpy.complex64: 'float2'
		}

		if self._dim > 1:
			self._tr = transpose.Transpose(typenames[self._dtype])
			self._temp = gpuarray.GPUArray((self._x * self._y * self._z * self._batch,), dtype=self._dtype)

		self._xplan = get_1dplan((self._x,), batch=self._y * self._z * self._batch)

		if self._dim > 1:
			self._yplan = get_1dplan((self._y,), batch=self._x * self._z * self._batch)

		if self._dim > 2:
			self._zplan = get_1dplan((self._z,), batch=self._x * self._y * self._batch)


	def _execute1d(self, idata, odata, direction):
		_fft(direction, idata, odata, plan=self._xplan)

	def _execute2d(self, idata, odata, direction):
		self._execute1d(idata, odata, direction)
		self._tr(self._temp.gpudata, odata.gpudata, self._x, self._y, self._batch * self._z)
		_fft(direction, self._temp, plan=self._yplan)
		self._tr(odata.gpudata, self._temp.gpudata, self._y, self._x, self._batch * self._z)

	def _execute3d(self, idata, odata, direction):
		self._execute2d(idata, odata, direction)
		self._tr(self._temp.gpudata, odata.gpudata, self._x * self._y, self._z, self._batch)
		_fft(direction, self._temp, plan=self._zplan)
		self._tr(odata.gpudata, self._temp.gpudata, self._z, self._x * self._y, self._batch)

	def execute(self, idata, odata, inverse=False):
		direction = CUFFT_INVERSE if inverse else CUFFT_FORWARD
		if self._dim == _FFT_1D:
			self._execute1d(idata, odata, direction)
		elif self._dim == _FFT_2D:
			self._execute2d(idata, odata, direction)
		else:
			self._execute3d(idata, odata, direction)
