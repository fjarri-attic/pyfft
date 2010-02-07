clFFT_Forward	= 	-1
clFFT_Inverse	= 	 1

clFFT_1D	= 0
clFFT_2D	= 1
clFFT_3D	= 3

class clFFT_Dim3:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

class clFFT_Complex:
	def __init__(self, real, imag):
		self.real = real
		self.imag = imag
