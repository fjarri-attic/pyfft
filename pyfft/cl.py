"""
OpenCL-specific part of the module.
"""

import pyopencl as cl
import warnings

if cl.VERSION < (0, 92):
	raise ImportError("PyOpenCL 0.92 or newer is required")

import sys
import numpy

from .plan import FFTPlan
from .kernel_helpers import log2


class Function:
	"""Wrapper for kernel function"""

	def __init__(self, context, program, name, split, block_size):
		self._program = program
		self._kernel = getattr(program, name)
		self._block_size = (block_size,)
		self._split = split
		self._context = context

	def isExecutable(self):
		# checks that number of registers this kernel uses is not
		# too big for requested block size
		return self._kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE,
			self._context.device) >= self._block_size[0]

	def prepare(self, grid, batch_size):
		grid_width, grid_height = grid
		self._global_size = (grid_width * grid_height * self._block_size[0],)
		self._batch_size = numpy.int32(batch_size) # CL kernel wrapper requires sized integer

	def __call__(self, queue, *args):
		kernel = self._kernel

		if self._split:
			self._kernel(queue, self._global_size, self._block_size,
				args[0], args[1], args[2], args[3], self._batch_size)
		else:
			self._kernel(queue, self._global_size, self._block_size,
				args[0], args[1], self._batch_size)


class Module:
	"""Wrapper for OpenCL module"""

	def __init__(self, context, kernel_string, fast_math, compiler_output=False):
		# OpenCL compiler can be a bit noisy sometimes
		# Kernel code does not normally produce any warnings, so I can safely
		# disable any compiler output by default
		if not compiler_output:
			obj = warnings.catch_warnings()
			warnings.simplefilter("ignore")

		# Casting source code to ASCII explicitly
		# New versions of Mako produce Unicode output by default,
		# and it makes OpenCL compiler unhappy
		self._program = cl.Program(context.context, str(kernel_string)).build(
			options=("-cl-mad-enable -cl-fast-relaxed-math" if fast_math else ""))

		if not compiler_output:
			del obj

		self._context = context

	def getFunction(self, name, split, block_size):
		return Function(self._context, self._program, name, split, block_size)


class Context:
	"""Class for plan execution context"""

	def __init__(self, context_obj, queue_obj):

		self._queue = queue_obj
		self.context = context_obj
		self.device = self._queue.device

		# TODO: find a way to get memory coalescing width and shared memory
		# granularity from device
		# Note: in PyCuda's DeviceData they are actually hardcoded too
		self.min_mem_coalesce_width = {4: 16, 8: 16, 16: 8}
		self.num_smem_banks = 16

		# TODO: I did not find any way of getting the maximum number of workgroups
		# We'll see if there are any problems with that
		self.max_grid_x = sys.maxint
		self.max_grid_y = 1

		self.max_shared_mem = self.device.get_info(cl.device_info.LOCAL_MEM_SIZE)

		workgroup_sizes = self.device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)
		self.max_block_size = workgroup_sizes[0]

	def allocate(self, size):
		return cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size=size)

	def compile(self, kernel_string, fast_math, compiler_output=False):
		return Module(self, kernel_string, fast_math, compiler_output=compiler_output)

	def createQueue(self):
		pass

	def getQueue(self):
		return self._queue

	def wait(self):
		self._queue.finish()

	def flush(self):
		self._queue.flush()

	def enqueue(self, func, *args):
		func(self._queue, *args)

	def isCuda(self):
		return False


def Plan(*args, **kwds):

	context_obj = kwds.pop('context', None)
	queue_obj = kwds.pop('queue', None)

	if queue_obj is not None:
		wait_for_finish = False
		context_obj = queue_obj.context
	elif context_obj is not None:
		queue_obj = cl.CommandQueue(context_obj)
		wait_for_finish = True
	else:
		raise ValueError("Either context or queue must be set")

	if 'wait_for_finish' not in kwds or kwds['wait_for_finish'] is None:
		kwds['wait_for_finish'] = wait_for_finish

	context = Context(context_obj, queue_obj)
	return FFTPlan(context, *args, **kwds)
