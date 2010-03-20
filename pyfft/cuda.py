"""
Cuda-specific part of the module
"""

from pycuda.driver import device_attribute
from pycuda.gpuarray import GPUArray
import pycuda.driver as cuda
from pycuda.tools import DeviceData
from pycuda.compiler import SourceModule

from .plan import FFTPlan
from .kernel_helpers import log2


class Function:
	"""Wrapper for kernel function"""

	def __init__(self, context, module, name, split, block_size):
		self._context = context
		self._module = module
		self._func_ref = self._module.get_function(name)
		arg_list = "PPPPi" if split else "PPi"
		self._block_size = block_size
		self._func_ref.prepare(arg_list, block=(block_size, 1, 1))
		self._split = split

	def prepare(self, grid, batch_size):
		self._grid = grid
		self._batch_size = batch_size

	def __call__(self, stream, *args):
		args = list(args)
		for i, arg in enumerate(args):
			if isinstance(arg, GPUArray):
				args[i] = arg.gpudata

		if self._split:
			self._func_ref.prepared_async_call(self._grid, stream, args[0], args[1], args[2], args[3], self._batch_size)
		else:
			self._func_ref.prepared_async_call(self._grid, stream, args[0], args[1], self._batch_size)

	def isExecutable(self):
		return self._func_ref.num_regs * self._block_size <= self._context.max_registers


class Module:
	"""Wrapper for Cuda SourceModule"""

	def __init__(self, context, kernel_string):
		self._module = SourceModule(kernel_string, no_extern_c=True)
		self._context = context

	def getFunction(self, name, split, block_size):
		return Function(self._context, self._module, name, split, block_size)


class Context:
	"""Class for plan execution context"""

	def __init__(self, device, stream, mempool):

		self._stream = stream
		self._recreate_stream = stream is None

		devdata = DeviceData(device)

		self.min_mem_coalesce_width = {}
		for size in [4, 8, 16]:
			self.min_mem_coalesce_width[size] = devdata.align_words(word_size=size)

		self.num_smem_banks = devdata.smem_granularity
		self.max_registers = device.get_attribute(device_attribute.MAX_REGISTERS_PER_BLOCK)
		self.max_grid_x = 2 ** log2(device.get_attribute(device_attribute.MAX_GRID_DIM_X))
		self.max_block_size = device.get_attribute(device_attribute.MAX_BLOCK_DIM_X)

		if mempool is None:
			self.allocate = cuda.mem_alloc
		else:
			self._mempool = mempool
			self.allocate = mempool.allocate

	def compile(self, kernel_string):
		return Module(self, kernel_string)

	def createQueue(self):
		if self._recreate_stream:
			self._stream = cuda.Stream()

	def wait(self):
		self._stream.synchronize()
		if self._recreate_stream:
			del self._stream

	def getQueue(self):
		return self._stream

	def enqueue(self, func, *args):
		func(self._stream, *args)

	def isCuda(self):
		return True


def Plan(*args, **kwds):
	mempool = kwds.pop('mempool', None)
	context_obj = kwds.pop('context', None)
	stream_obj = kwds.pop('stream', None)

	if stream_obj is not None:
		device = cuda.Context.get_device()
		wait_for_finish = False
	elif context_obj is not None:
		device = context_obj.get_device()
		wait_for_finish = True
		stream_obj = None
	else:
		device = cuda.Context.get_device()
		stream_obj = cuda.Stream()
		wait_for_finish = True

	if 'wait_for_finish' not in kwds or kwds['wait_for_finish'] is None:
		kwds['wait_for_finish'] = wait_for_finish

	context = Context(device, stream_obj, mempool)

	return FFTPlan(context, *args, **kwds)
