cl_fft_kernel_x = 0
cl_fft_kernel_y = 1
cl_fft_kernel_z = 2

class cl_fft_kernel_info:
	def __init__(self):
		self.cl_kernel = None
		self.kernel_name = ""
		self.lmem_size = 0
		self.num_workgroups = 0
		self.num_workitems_per_workgroup = 0
		self.cl_fft_kernel_dir = None
		self.in_place_possible = None

# TODO: think of something more effective
def log2(n):
	pos = 0
	for pow in [16, 8, 4, 2, 1]:
		if n >= 2 ** pow:
			n /= (2 ** pow)
			pos += pow
	return pos
