from pycuda.driver import device_attribute
from pycuda.gpuarray import GPUArray
import pycuda.driver as cuda
from pycuda.tools import DeviceData

from plan import FFTPlan

class Devdata:

	def __init__(self, device):
		devdata = DeviceData(device)

		self.min_mem_coalesce_width = {}
		for size in [4, 8, 16]:
			self.min_mem_coalesce_width[size] = devdata.align_bytes(word_size=size) / size

		self.num_smem_banks = devdata.smem_granularity
		self.max_registers = device.get_attribute(device_attribute.MAX_REGISTERS_PER_BLOCK)
		self.max_grid_x = 2 ** log2(device.get_attribute(device_attribute.MAX_GRID_DIM_X))
		self.max_block_size = device.get_attribute(device_attribute.MAX_BLOCK_DIM_X)

class Plan:

	def __init__(self, *args, **kwds):
		return FFTPlan(*args, **kwds, std_alloc=cuda.mem_alloc,
			devdata=Devdata(), compiler=Compiler())