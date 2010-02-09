#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cufft.h>
#include <cutil_inline.h>

#include <cudabuffer.h>
#include <batchfft.h>

// Increase this if your videocards allows you to do it
// Must be a power of two
#define TEST_BUFFER_SIZE (16 * 1024 * 1024)

#define NUMITER 10

void runTest(int x, int y, int z)
{
	unsigned int timer = 0;
	cutilCheckError(cutCreateTimer(&timer));

	int batch = TEST_BUFFER_SIZE / (x * y * z * sizeof(cufftComplex));
	if(batch == 0)
	{
		printf("Resulting buffer size is too big, test skipped\n");
		return;
	}

	bool use_batchfft;
	cufftHandle cufft_plan;
	batchfftHandle batchfft_plan;
	CudaBuffer<cufftComplex> idata(x * y * z * batch), odata(x * y * z * batch);

	printf("--- (%d, %d, %d), batch %d\n", x, y, z, batch);
	float gflop = 5.0e-9 * log2((float)(x * y * z)) * x * y * z * batch;

	if(y == 1 && z == 1)
		use_batchfft = false;
	else
		use_batchfft = true;

	// prepare plans
	if(use_batchfft)
		if(z != 1)
			cufftSafeCall(batchfftPlan3d(&batchfft_plan, z, y, x, CUFFT_C2C, batch));
		else
			cufftSafeCall(batchfftPlan2d(&batchfft_plan, y, x, CUFFT_C2C, batch));
	else
		cufftSafeCall(cufftPlan1d(&cufft_plan, x, CUFFT_C2C, batch));

	// Warming up
	if(use_batchfft)
		cufftSafeCall(batchfftExecute(batchfft_plan, (cufftComplex*)idata, (cufftComplex*)odata, CUFFT_FORWARD));
	else
		cufftSafeCall(cufftExecC2C(cufft_plan, (cufftComplex*)idata, (cufftComplex*)odata, CUFFT_FORWARD));
	cutilSafeCall(cudaThreadSynchronize());

	// measure out of place time
	cutilCheckError(cutStartTimer(timer));
	for(int i = 0; i < NUMITER; i++)
		if(use_batchfft)
			cufftSafeCall(batchfftExecute(batchfft_plan, (cufftComplex*)idata, (cufftComplex*)odata, CUFFT_FORWARD));
		else
			cufftSafeCall(cufftExecC2C(cufft_plan, (cufftComplex*)idata, (cufftComplex*)odata, CUFFT_FORWARD));
	cutilSafeCall(cudaThreadSynchronize());
	cutilCheckError(cutStopTimer(timer));
	printf("Out-of-place time: %f ms (%f GFLOPS)\n",
	       cutGetTimerValue(timer) / NUMITER,
	       gflop / (cutGetTimerValue(timer) / NUMITER / 1000));

	cutilCheckError(cutResetTimer(timer));

	// measure inplace
	cutilCheckError(cutStartTimer(timer));
	for(int i = 0; i < NUMITER; i++)
		if(use_batchfft)
			cufftSafeCall(batchfftExecute(batchfft_plan, (cufftComplex*)idata, (cufftComplex*)idata, CUFFT_FORWARD));
		else
			cufftSafeCall(cufftExecC2C(cufft_plan, (cufftComplex*)idata, (cufftComplex*)idata, CUFFT_FORWARD));
	cutilSafeCall(cudaThreadSynchronize());
	cutilCheckError(cutStopTimer(timer));
	printf("Inplace time: %f ms (%f GFLOPS)\n",
	       cutGetTimerValue(timer) / NUMITER,
	       gflop / (cutGetTimerValue(timer) / NUMITER / 1000));

	cutilCheckError( cutDeleteTimer( timer));

	if(use_batchfft)
		batchfftDestroy(batchfft_plan);
	else
		cufftDestroy(cufft_plan);
}

int main(int argc, char** argv)
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

	// 1D
	runTest(16, 1, 1);
	runTest(1024, 1, 1);
	runTest(8192, 1, 1);

	// 2D
	runTest(16, 16, 1);
	runTest(128, 128, 1);
	runTest(1024, 1024, 1);

	// 3D
	runTest(16, 16, 16);
	runTest(32, 32, 128);
	runTest(128, 128, 128);
}
