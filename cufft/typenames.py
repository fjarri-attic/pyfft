import numpy

class _Type:
	def __init__(self, c_name, numpy_class):
		self.c_name = c_name
		self.numpy_class = numpy_class

		inst = numpy_class()
		self.dtype = inst.dtype
		self.nbytes = inst.nbytes

float32 = _Type('float', numpy.float32)
float64 = _Type('double', numpy.float64)
complex32 = _Type('float2', numpy.complex64)
