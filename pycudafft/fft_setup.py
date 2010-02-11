from fft_internal import *
from fft_kernelstring import *
from fft_base_kernels import *
from clFFT import *
from pycuda.autoinit import device
from pycuda.compiler import SourceModule
from pycuda.driver import device_attribute


def getBlockConfigAndKernelString(plan):

	plan.temp_buffer_needed = 0;
	plan.kernel_string = base_kernels.render()

	plan.kernel_string += "\nextern \"C\"{\n"

	if plan.dim == clFFT_1D:
		plan.kernel_string += FFT1D(plan, cl_fft_kernel_x)
	elif plan.dim == clFFT_2D:
		plan.kernel_string += FFT1D(plan, cl_fft_kernel_x)
		plan.kernel_string += FFT1D(plan, cl_fft_kernel_y)
	elif plan.dim == clFFT_3D:
		plan.kernel_string += FFT1D(plan, cl_fft_kernel_x)
		plan.kernel_string += FFT1D(plan, cl_fft_kernel_y)
		plan.kernel_string += FFT1D(plan, cl_fft_kernel_z)
	else:
		raise Exception("Wrong dimension")

	plan.kernel_string += "\n}\n"

	plan.temp_buffer_needed = False
	for kInfo in plan.kernel_info:
		if not kInfo.in_place_possible:
			plan.temp_buffer_needed = True

def createKernelList(plan):
	shared_mem_limit = device.get_attribute(device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
	reg_limit = device.get_attribute(device_attribute.MAX_REGISTERS_PER_BLOCK)

	for kInfo in plan.kernel_info:
		kInfo.function_ref = plan.module.get_function(kInfo.kernel_name)
		if kInfo.function_ref.shared_size_bytes > shared_mem_limit:
			raise Exception("Insufficient shared memory")
		if kInfo.function_ref.num_regs * kInfo.num_workitems_per_workgroup > reg_limit:
			raise Exception("Insufficient registers")

def getMaxKernelWorkGroupSize(plan):
	# TODO: investigate the original function and write proper port
	return 32768


class FFTPlan:

	def __init__(self, x, y, z, dim, data_format):

		# TODO: check that dimensions are the power of two
		# and number of elements in n corresponds to dim

		class _Dim:
			def __init__(self, x, y, z):
				self.x = x
				self.y = y
				self.z = z

		self.dataFormat = data_format
		self.n = _Dim(x, y, z)
		self.dim = dim
		self.num_kernels = 0
		self.program = 0
		self.temp_buffer_needed = False
		self.last_batch_size = 0
		self.tempmemobj = None
		self.tempmemobj_re = None
		self.tempmemobj_im = None
		self.max_localmem_fft_size = 2048
		self.max_radix = 16
		self.min_mem_coalesce_width = 16

		# TODO: get this parameter properly from device instead of calculating it
		self.num_local_mem_banks = device.get_attribute(device_attribute.WARP_SIZE) / 2

		self.max_work_item_per_workgroup = device.get_attribute(device_attribute.MAX_BLOCK_DIM_X)

		# TODO: make this 'recompile-if-necessary' code more good looking
		done = False
		while not done:
			self.kernel_string = ""
			self.kernel_info = []
			getBlockConfigAndKernelString(self)
			self.module = SourceModule(self.kernel_string, no_extern_c=True)
			try:
				createKernelList(self)
			except:
				if self.max_work_item_per_workgroup > 1:
					self.max_work_item_per_workgroup /= 2
					continue
				raise Exception("Cannot meet number of registers/shared memory requirements")
			done = True
