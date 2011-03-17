#ifndef _BUFFER_H
#define _BUFFER_H

#include <assert.h>
#include <cutil_inline.h>
#include <stdio.h>
#include <stdlib.h>

// Wrapper for cuda memory buffer (just some convenience functions)
template<class T>
class CudaBuffer
{
private:
	T *buffer;
	size_t length;

	// block copy constructor, just in case
	CudaBuffer(const CudaBuffer<T> &inst) {}

	void operator=(const CudaBuffer&) {}

public:
	CudaBuffer()
	{
		length = 0;
		buffer = NULL;
	}

	CudaBuffer(size_t len)
	{
		buffer = NULL;
		init(len);
	}

	~CudaBuffer()
	{
		if(buffer != NULL)
			release();
	}

	void init(size_t len)
	{
		assert(buffer == NULL);
		length = len;
		cutilSafeCall(cudaMalloc((void**)&buffer, length * sizeof(T)));

		// Fill array with random numbers
		// To avoid NaNs, which can slow down kernel execution in tests
		// (now I can use newly created CudaBuffers in performance tests
		// without explicitly filling them with data)
		T* contents = new T[length];
		for(int i = 0; i < length; i++)
		{
			float rnd_num = (float)rand() / RAND_MAX;
			contents[i].x = rnd_num;
			contents[i].y = rnd_num;
		}

		copyFrom(contents);

		delete[] contents;
	}

	void release()
	{
		assert(buffer != NULL);
		cutilSafeCall(cudaFree(buffer));
		buffer = NULL;
		length = 0;
	}

	size_t len() const
	{
		return length;
	}

	void copyFrom(const T *h_data, size_t len = 0)
	{
		if(len == 0)
			len = length;
		assert(len <= length);
		cutilSafeCall(cudaMemcpy(buffer, h_data, len * sizeof(T), cudaMemcpyHostToDevice));
	}

	void copyTo(T *h_data, size_t len = 0) const
	{
		if(len == 0)
			len = length;
		assert(len <= length);
		cutilSafeCall(cudaMemcpy(h_data, buffer, len * sizeof(T), cudaMemcpyDeviceToHost));
	}

	void copyFrom(const CudaBuffer<T> &other, size_t len = 0, size_t other_offset = 0)
	{
		if(len == 0)
			len = other.len();
		assert(len + other_offset <= other.len());
		assert(len <= length);
		cutilSafeCall(cudaMemcpy(buffer, other + other_offset, len * sizeof(T), cudaMemcpyDeviceToDevice));
	}

	void copyTo(CudaBuffer<T> &other, size_t len = 0, size_t other_offset = 0) const
	{
		if(len == 0)
			len = length;
		assert(len + other_offset <= other.len());
		assert(len <= length);
		cutilSafeCall(cudaMemcpy(other + other_offset, buffer, len * sizeof(T), cudaMemcpyDeviceToDevice));
	}

	operator T*() const
	{
		return buffer;
	}
};

#endif
