#ifndef _TRANSPOSE_CUH
#define _TRANSPOSE_CUH

// size of half-warp (for kernel optimizations)
#define HALF_WARP_SIZE 16


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
template<class T>
__global__ void transposeKernel(T* odata, const T* idata, int width, int height, int num)
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
	__shared__ T block[(HALF_WARP_SIZE + 1) * HALF_WARP_SIZE];

	unsigned int xBlock = __umul24(HALF_WARP_SIZE, blockIdx.x);
	unsigned int yBlock = __umul24(HALF_WARP_SIZE, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;
	unsigned int size = __umul24(width, height);
	unsigned int index_block = __umul24(threadIdx.y, HALF_WARP_SIZE + 1) + threadIdx.x;
	unsigned int index_transpose = __umul24(threadIdx.x, HALF_WARP_SIZE + 1) + threadIdx.y;
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


/**
 * Fast matrix transpose function
 *
 * @param odata Output buffer for transposed batch of matrices, must be different than idata
 * @param idata Input batch of matrices
 * @param width Width of each matrix, must be a multiple of HALF_WARP_SIZE
 * @param height Height of each matrix, must be a multiple of HALF_WARP_SIZE
 * @param num Matrices in the batch
 *
 * @return Cuda error code
 */
template<class T>
cudaError_t transpose(T* odata, const T* idata, int width, int height, int num)
{
	if(width % HALF_WARP_SIZE || height % HALF_WARP_SIZE)
		return cudaErrorInvalidConfiguration;

	// FIXME: add support for cases when (width / HALF_WARP_SIZE) is larger than maximum grid size
	dim3 grid(width / HALF_WARP_SIZE, height / HALF_WARP_SIZE, 1);
	dim3 block(HALF_WARP_SIZE, HALF_WARP_SIZE, 1);

	transposeKernel<T><<<grid, block, sizeof(T) * (HALF_WARP_SIZE + 1) * HALF_WARP_SIZE>>>(odata, idata, width, height, num);

	return cudaGetLastError();
}

#endif
