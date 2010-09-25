#ifndef _DEFINES_H_
#define _DEFINES_H_

#include <cufft.h>

// Increase this if your videocards allows you to do it
// Must be a power of two
#define TEST_BUFFER_SIZE (32 * 1024 * 1024)

#define NUMITER 10

//#define DOUBLE

#ifdef DOUBLE

typedef cufftDoubleComplex complexType;
#define PLAN_TYPE CUFFT_Z2Z
#define executePlan cufftExecZ2Z

#else

typedef cufftComplex complexType;
#define PLAN_TYPE CUFFT_C2C
#define executePlan cufftExecC2C

#endif

#endif
