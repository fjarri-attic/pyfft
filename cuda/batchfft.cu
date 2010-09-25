#include <cutil_inline.h>
#include <assert.h>

#include "defines.h"
#include "batchfft.h"
#include "transpose.cuh"

////////////////////////////////////////////////////////////////////////////////
cufftResult batchfftFillPlan(batchfftHandle *plan, int nx, int ny, int nz, cufftType type, int batch)
{
	if(type != PLAN_TYPE)
		return CUFFT_INVALID_TYPE;

	if(nx % HALF_WARP_SIZE != 0)
		return CUFFT_INVALID_SIZE;

	if(ny % HALF_WARP_SIZE != 0)
		return CUFFT_INVALID_SIZE;

	if(nz != 1 && nz % HALF_WARP_SIZE != 0)
		return CUFFT_INVALID_SIZE;

	if(nz == 1)
	{
		plan->dim = 2;

		// Swap nx and ny so they correspoind to the 2D CUFFT API.
		// 2D cufft expects them in the order for a declared C array:
		//
		// complexType array[nx][ny];
		// cufftPlan2d(plan, nx, ny, type);
		//
		// even though ny would be considered the "x" array index for row-major
		// array storage.
		plan->ny = nx;
		plan->nx = ny;
		plan->nz = 1;
	}
	else
	{
		plan->dim = 3;

		// Swap dimensions, the reason is the same as for 2D case.
		plan->nx = nz;
		plan->ny = ny;
		plan->nz = nx;
	}

	plan->type = type;
	plan->batch = batch;

	cufftResult ret = CUFFT_SUCCESS;
	cudaError_t cudaret = cudaSuccess;

	cudaret = cudaMalloc(&(plan->temp), plan->nx * plan->ny * plan->nz * plan->batch * sizeof(complexType));
	if(cudaret != cudaSuccess)
		return CUFFT_ALLOC_FAILED;

	ret = cufftPlan1d(&(plan->xplan), plan->nx, plan->type, plan->ny * plan->nz * plan->batch);
	if(ret != CUFFT_SUCCESS)
	{
		cudaFree(plan->temp);
		plan->temp = NULL;
		return ret;
	}

	ret = cufftPlan1d(&(plan->yplan), plan->ny, plan->type, plan->nx * plan->nz * plan->batch);
	if(ret != CUFFT_SUCCESS)
	{
		cudaFree(plan->temp);
		plan->temp = NULL;
		cufftDestroy(plan->xplan);
		return ret;
	}

	if(plan->dim == 3)
	{
		ret = cufftPlan1d(&(plan->zplan), plan->nz, plan->type, plan->nx * plan->ny * plan->batch);
		if(ret != CUFFT_SUCCESS)
		{
			cudaFree(plan->temp);
			plan->temp = NULL;
			cufftDestroy(plan->xplan);
			cufftDestroy(plan->yplan);
			return ret;
		}
	}

	return CUFFT_SUCCESS;
}

cufftResult batchfftPlan2d(batchfftHandle* plan, int nx, int ny, cufftType type, int batch)
{
	return batchfftFillPlan(plan, nx, ny, 1, type, batch);
}

cufftResult batchfftPlan3d(batchfftHandle* plan, int nx, int ny, int nz, cufftType type, int batch)
{
	return batchfftFillPlan(plan, nx, ny, nz, type, batch);
}

////////////////////////////////////////////////////////////////////////////////
cufftResult batchfftDestroy(batchfftHandle &plan)
{
	assert(plan.temp != NULL);

	cufftDestroy(plan.xplan);
	cufftDestroy(plan.yplan);
	if(plan.dim == 3)
		cufftDestroy(plan.zplan);
	cudaFree(plan.temp);
	plan.temp = NULL;

	return CUFFT_SUCCESS;
}

cufftResult batchfftExecute2D(batchfftHandle &plan, complexType* idata, complexType* odata, int sign)
{
	cufftResult cufftret = CUFFT_SUCCESS;
	cudaError_t cudaret = cudaSuccess;

	// Transform rows
	cufftret = executePlan(plan.xplan, idata, odata, sign);
	if(cufftret != CUFFT_SUCCESS)
		return cufftret;

	// Transpose
	cudaret = transpose(plan.temp, odata, plan.nx, plan.ny, plan.batch * plan.nz);
	if(cudaret != cudaSuccess)
		return CUFFT_EXEC_FAILED;

	// Transform columns
	cufftret = executePlan(plan.yplan, plan.temp, plan.temp, sign);
	if(cufftret != CUFFT_SUCCESS)
		return cufftret;

	// Transpose back
	cudaret = transpose(odata, plan.temp, plan.ny, plan.nx, plan.batch * plan.nz);
	if(cudaret != cudaSuccess)
		return CUFFT_EXEC_FAILED;

	return CUFFT_SUCCESS;
}

cufftResult batchfftExecute3D(batchfftHandle &plan, complexType* idata, complexType* odata, int sign)
{
	cufftResult cufftret = CUFFT_SUCCESS;
	cudaError_t cudaret = cudaSuccess;

	cufftret = batchfftExecute2D(plan, idata, odata, sign);
	if(cufftret != CUFFT_SUCCESS)
		return cufftret;

	cudaret = transpose(plan.temp, odata, plan.nx * plan.ny, plan.nz, plan.batch);
	if(cudaret != cudaSuccess)
		return CUFFT_EXEC_FAILED;

	cufftret = executePlan(plan.zplan, plan.temp, plan.temp, sign);
	if(cufftret != CUFFT_SUCCESS)
		return cufftret;

	cudaret = transpose(odata, plan.temp, plan.nz, plan.nx * plan.ny, plan.batch);
	if(cudaret != cudaSuccess)
		return CUFFT_EXEC_FAILED;

	return CUFFT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
cufftResult batchfftExecute(batchfftHandle &plan, complexType* idata, complexType* odata, int sign)
{
	if(plan.dim == 2)
		return batchfftExecute2D(plan, idata, odata, sign);
	else if(plan.dim == 3)
		return batchfftExecute3D(plan, idata, odata, sign);
	else
		return CUFFT_INVALID_PLAN;
}
