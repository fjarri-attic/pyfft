#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cufft.h>
#include <cutil_inline.h>

#include <cudabuffer.h>
#include <defines.h>


void runTest(int x, int y, int z)
{
	unsigned int timer = 0;
	cutilCheckError(cutCreateTimer(&timer));

	int batch = TEST_BUFFER_SIZE / (x * y * z * sizeof(complexType));
	if(batch == 0)
	{
		printf("Resulting buffer size is too big, test skipped\n");
		return;
	}

	cufftHandle plan;
	CudaBuffer<complexType> idata(x * y * z * batch), odata(x * y * z * batch);

	printf("--- (%d, %d, %d), batch %d\n", x, y, z, batch);
	float gflop = 5.0e-9 * log2((float)(x * y * z)) * x * y * z * batch;

	// prepare plan
	int n[3] = {x, y, z};
	int rank = 1;
	if(y != 1) rank = 2;
	if(z != 1) rank = 3;

	cufftSafeCall(cufftPlanMany(&plan, rank, n, NULL, 1, 0, NULL, 1, 0, PLAN_TYPE, batch));

	cufftSafeCall(executePlan(plan, (complexType*)idata, (complexType*)odata, CUFFT_FORWARD));
	cutilSafeCall(cudaThreadSynchronize());

	// measure out of place time
	cutilCheckError(cutStartTimer(timer));
	for(int i = 0; i < NUMITER; i++)
		cufftSafeCall(executePlan(plan, (complexType*)idata, (complexType*)odata, CUFFT_FORWARD));

	cutilSafeCall(cudaThreadSynchronize());
	cutilCheckError(cutStopTimer(timer));
	printf("Out-of-place time: %f ms (%f GFLOPS)\n",
	       cutGetTimerValue(timer) / NUMITER,
	       gflop / (cutGetTimerValue(timer) / NUMITER / 1000));

	cutilCheckError(cutResetTimer(timer));

	// measure inplace
	cutilCheckError(cutStartTimer(timer));
	for(int i = 0; i < NUMITER; i++)
		cufftSafeCall(executePlan(plan, (complexType*)idata, (complexType*)idata, CUFFT_FORWARD));

	cutilSafeCall(cudaThreadSynchronize());
	cutilCheckError(cutStopTimer(timer));
	printf("Inplace time: %f ms (%f GFLOPS)\n",
	       cutGetTimerValue(timer) / NUMITER,
	       gflop / (cutGetTimerValue(timer) / NUMITER / 1000));

	cutilCheckError( cutDeleteTimer( timer));

	cufftDestroy(plan);
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
	runTest(8, 8, 64);
	runTest(16, 16, 16);
	runTest(16, 16, 128);
	runTest(32, 32, 128);
	runTest(128, 128, 128);
}
