cl_fft_kernel_x = 0
cl_fft_kernel_y = 1
cl_fft_kernel_z = 2

# TODO: think of something more effective
def log2(n):
	pos = 0
	for pow in [16, 8, 4, 2, 1]:
		if n >= 2 ** pow:
			n /= (2 ** pow)
			pos += pow
	return pos
