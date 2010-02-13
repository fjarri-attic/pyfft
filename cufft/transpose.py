from mako.template import Template
from pycuda.autoinit import device
from pycuda.compiler import SourceModule
from pycuda.driver import device_attribute

_kernel_template = Template("""
/**
 * Fast matrix transpose kernel
 *
 * Uses shared memory to coalesce global memory reads and writes, improving performance.
 *
 * @param odata Output buffer for transposed batch of matrices, must be different than idata
 * @param idata Input batch of matrices
 * @param width Width of each matrix, must be a multiple of HALF_WARP_SIZE
 * @param height Height of each matrix, must be a multiple of HALF_WARP_SIZE
 * @param num Matrices in the batch
 */
__global__ void transposeKernel(${typename}* odata, const ${typename}* idata, int width, int height, int num)
{
	// To prevent shared memory bank confilcts:
	// - Load each component into a different array. Since the array size is a
	//   multiple of the number of banks (16), each thread reads x and y from
	//   the same bank. If a single value_pair array is used, thread n would read
	//   x and y from banks n and n+1, and thread n+8 would read values from the
	//   same banks - causing a bank conflict.
	// - Use HALF_WARP_SIZE+1 as the x size of the array. This way each row of the
	//   array starts in a different bank - so reading from shared memory
	//   doesn't cause bank conflicts when writing the transpose out to global
	//   memory.
	__shared__ ${typename} block[(${half_warp_size} + 1) * ${half_warp_size}];

	unsigned int xBlock = __umul24(${half_warp_size}, blockIdx.x);
	unsigned int yBlock = __umul24(${half_warp_size}, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;
	unsigned int size = __umul24(width, height);
	unsigned int index_block = __umul24(threadIdx.y, ${half_warp_size} + 1) + threadIdx.x;
	unsigned int index_transpose = __umul24(threadIdx.x, ${half_warp_size} + 1) + threadIdx.y;
	unsigned int index_in = __umul24(width, yIndex) + xIndex;
	unsigned int index_out = __umul24(height, xBlock + threadIdx.y) + yBlock + threadIdx.x;

	for(int n = 0; n < num; ++n)
	{
		block[index_block] = idata[index_in];

		__syncthreads();

		odata[index_out] = block[index_transpose];

		index_in += size;
		index_out += size;
	}
}
""")

class Transpose:

	def __init__(self, typename):

		self._half_warp_size = device.get_attribute(device_attribute.WARP_SIZE) / 2

		# render function from template
		source = _kernel_template.render(typename=typename, half_warp_size=self._half_warp_size)

		# get function from module
		_kernel_module = SourceModule(source)
		self._func = _kernel_module.get_function("transposeKernel")

		# prepare function call
		block = (self._half_warp_size, self._half_warp_size, 1)
		self._func.prepare("PPiii", block=block)

	def __call__(self, odata, idata, width, height, num):
		"""
		Fast matrix transpose function
		odata: Output buffer for transposed batch of matrices, must be different than idata
		idata: Input batch of matrices
		width: Width of each matrix, must be a multiple of HALF_WARP_SIZE
		height: Height of each matrix, must be a multiple of HALF_WARP_SIZE
		num: number of matrices in the batch
		"""
		assert width % self._half_warp_size == 0
		assert height % self._half_warp_size == 0

		grid = (width / self._half_warp_size, height / self._half_warp_size)
		self._func.prepared_call(grid, odata, idata, width, height, num)
