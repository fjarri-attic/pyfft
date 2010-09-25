#ifndef _BATCHFFT_H_
#define _BATCHFFT_H_

#include "defines.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Batched 2D FFT plan
 *
 * This struct is meant to be opaque to the caller -
 * it is initialized by batchfftPlan2d() and destroyed
 * by batchfftDestroy().
 */
struct batchfftHandle
{
	int dim;
	int nx;		///< FFT size dimension
	int ny;		///< FFT size dimension
	int nz;
	int batch;	///< Number of FFTs in the batch

	cufftType type;	///< FFT type

	cufftHandle xplan;	///< CUFFT plan for FFTing rows
	cufftHandle yplan;	///< CUFFT plan for FFTing columns
	cufftHandle zplan;	///< CUFFT plan for FFTing columns

	complexType *temp;	///< Temporary buffer for transpose kernel

	batchfftHandle() : temp(NULL) { }; // use temp to tell if this handle has been allocated
};

/**
 * Create a batched 2D FFT plan
 *
 * This implementation requires a temporary buffer on the GPU the same size as
 * the data to transform. The buffer is allocated when the plan is created and
 * released when the plan is destroyed.
 *
 * @param plan Pointer to an uninitialized plan
 * @param nx Dimension, must be > 1 and a multiple of 16
 * @param ny Dimension, must be > 1 and a multiple of 16
 * @param type FFT type (only CUFFT_C2C supported)
 * @param batch Number of FFTs in the batch
 *
 * @returns See CUFFT documentation for possible return values
 */
cufftResult batchfftPlan2d(batchfftHandle* plan, int nx, int ny, cufftType type, int batch);

/**
 * Create a batched 3D FFT plan
 *
 * This implementation requires a temporary buffer on the GPU the same size as
 * the data to transform. The buffer is allocated when the plan is created and
 * released when the plan is destroyed.
 *
 * @param plan Pointer to an uninitialized plan
 * @param nx Dimension, must be > 1 and a multiple of 16
 * @param ny Dimension, must be > 1 and a multiple of 16
 * @param nz Dimension, must be > 1 and a multiple of 16
 * @param type FFT type (only CUFFT_C2C supported)
 * @param batch Number of FFTs in the batch
 *
 * @returns See CUFFT documentation for possible return values
 */
cufftResult batchfftPlan3d(batchfftHandle* plan, int nx, int ny, int nz, cufftType type, int batch);

/**
 * Destroy a batched FFT plan
 *
 * This implementation requires a temporary buffer on the GPU the same size as
 * the data to transform. The buffer is allocated when the plan is created and
 * released when the plan is destroyed.
 *
 * @param plan Plan to destroy
 *
 * @returns See CUFFT documentation for possible return values
 */
cufftResult batchfftDestroy(batchfftHandle &plan);

/**
 * Execute a batched 2D FFT
 *
 * @param plan Plan
 * @param idata Pointer to input data
 * @param odata Pointer to output data (if same as idata, performs in-place transforms)
 * @param sign CUFFT_FORWARD or CUFFT_INVERSE
 *
 * @returns See CUFFT documentation for possible return values
 */
cufftResult batchfftExecute(batchfftHandle &plan, complexType* idata, complexType* odata, int sign);

#ifdef __cplusplus
}
#endif

#endif
