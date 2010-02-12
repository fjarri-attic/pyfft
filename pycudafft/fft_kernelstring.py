import math
from fft_internal import *
from clFFT import *

# For any n, this function decomposes n into factors for loacal memory tranpose
# based fft. Factors (radices) are sorted such that the first one (radixArray[0])
# is the largest. This base radix determines the number of registers used by each
# work item and product of remaining radices determine the size of work group needed.
# To make things concrete with and example, suppose n = 1024. It is decomposed into
# 1024 = 16 x 16 x 4. Hence kernel uses float2 a[16], for local in-register fft and
# needs 16 x 4 = 64 work items per work group. So kernel first performance 64 length
# 16 ffts (64 work items working in parallel) following by transpose using local
# memory followed by again 64 length 16 ffts followed by transpose using local memory
# followed by 256 length 4 ffts. For the last step since with size of work group is
# 64 and each work item can array for 16 values, 64 work items can compute 256 length
# 4 ffts by each work item computing 4 length 4 ffts.
# Similarly for n = 2048 = 8 x 8 x 8 x 4, each work group has 8 x 8 x 4 = 256 work
# iterms which each computes 256 (in-parallel) length 8 ffts in-register, followed
# by transpose using local memory, followed by 256 length 8 in-register ffts, followed
# by transpose using local memory, followed by 256 length 8 in-register ffts, followed
# by transpose using local memory, followed by 512 length 4 in-register ffts. Again,
# for the last step, each work item computes two length 4 in-register ffts and thus
# 256 work items are needed to compute all 512 ffts.
# For n = 32 = 8 x 4, 4 work items first compute 4 in-register
# lenth 8 ffts, followed by transpose using local memory followed by 8 in-register
# length 4 ffts, where each work item computes two length 4 ffts thus 4 work items
# can compute 8 length 4 ffts. However if work group size of say 64 is choosen,
# each work group can compute 64/ 4 = 16 size 32 ffts (batched transform).
# Users can play with these parameters to figure what gives best performance on
# their particular device i.e. some device have less register space thus using
# smaller base radix can avoid spilling ... some has small local memory thus
# using smaller work group size may be required etc

def getRadixArray(n, maxRadix):
	if maxRadix > 1:
		maxRadix = min(n, maxRadix)
		radixArray = []
		while n > maxRadix:
			radixArray.append(maxRadix)
			n /= maxRadix
		radixArray.append(n)
		return radixArray

	if n in [2, 4, 8]:
		return [n]
	elif n in [16, 32, 64]:
		return [8, n / 8]
	elif n == 128:
		return [8, 4, 4]
	elif n == 256:
		return [4, 4, 4, 4]
	elif n == 512:
		return [8, 8, 8]
	elif n == 1024:
		return [16, 16, 4]
	elif n == 2048:
		return [8, 8, 8, 4]
	else:
		raise Exception("Wrong problem size: " + str(n))

def insertHeader(kernelName, dataFormat):
	if dataFormat == clFFT_SplitComplexFormat:
		return "__global__ void " + kernelName + "(float *in_real, float *in_imag, float *out_real, float *out_imag, int dir, int S)\n"
	else:
		return "__global__ void " + kernelName + "(float2 *in, float2 *out, int dir, int S)\n"

def insertVariables(maxRadix):
	# need to fill a[] with zeros, because otherwise nvcc crashes
	# (it considers a[] uninitalized)
	return """
	int i, j, r, indexIn, indexOut, index, tid, bNum, xNum, k, l;
	int s, ii, jj, offset;
	float2 w;
	float ang, angf, ang1;
	size_t lMemStore, lMemLoad;
	float2 a[""" + str(maxRadix) + """] = {""" + ','.join(['0'] * maxRadix * 2) + """};
	int lId = threadIdx.x;
	int groupId = blockIdx.x;
	"""

def formattedLoad(aIndex, gIndex, dataFormat):
	if dataFormat == clFFT_SplitComplexFormat:
		return "	a[" + str(aIndex) + "].x = in_real[" + str(gIndex) + "];\n" + \
			"	a[" + str(aIndex) + "].y = in_imag[" + str(gIndex) + "];\n"
	else:
		return "	a[" + str(aIndex) + "] = in[" + str(gIndex) + "];\n"

def formattedStore(aIndex, gIndex, dataFormat):
	if dataFormat == clFFT_SplitComplexFormat:
		return "	out_real[" + str(gIndex) + "] = a[" + str(aIndex) + "].x;\n" + \
			"	out_imag[" + str(gIndex) + "] = a[" + str(aIndex) + "].y;\n"
	else:
		return "	out[" + str(gIndex) + "] = a[" + str(aIndex) + "];\n"

def insertGlobalLoadsAndTranspose(N, numWorkItemsPerXForm, numXFormsPerWG, R0, mem_coalesce_width, dataFormat):
	res = ""
	log2NumWorkItemsPerXForm = log2(numWorkItemsPerXForm)
	groupSize = numWorkItemsPerXForm * numXFormsPerWG
	lMemSize = 0

	if numXFormsPerWG > 1:
		res += "	s = S & " + str(numXFormsPerWG - 1) + ";\n"

	if numWorkItemsPerXForm >= mem_coalesce_width:
		if numXFormsPerWG > 1:
			res += "	ii = lId & " + str(numWorkItemsPerXForm - 1) + ";\n"
			res += "	jj = lId >> " + str(log2NumWorkItemsPerXForm) + ";\n"
			res += "	if( !s || (groupId < gridDim.x - 1) || (jj < s) ) {\n"
			res += "		offset = mad24( mad24(groupId, " + \
				str(numXFormsPerWG) + ", jj), " + str(N) + ", ii );\n"

			if dataFormat == clFFT_InterleavedComplexFormat:
				res += "		in += offset;\n"
				res += "		out += offset;\n"
			else:
				res += "		in_real += offset;\n"
				res += "		in_imag += offset;\n"
				res += "		out_real += offset;\n"
				res += "		out_imag += offset;\n"

			for i in range(R0):
				res += formattedLoad(i, i*numWorkItemsPerXForm, dataFormat)
			res += "	}\n"
		else:
			res += "	ii = lId;\n"
			res += "	jj = 0;\n"
			res += "	offset =  mad24(groupId, " + str(N) + ", ii);\n"

			if dataFormat == clFFT_InterleavedComplexFormat:
				res += "		in += offset;\n"
				res += "		out += offset;\n"
			else:
				res += "		in_real += offset;\n"
				res += "		in_imag += offset;\n"
				res += "		out_real += offset;\n"
				res += "		out_imag += offset;\n"

			for i in range(R0):
				res += formattedLoad(i, i*numWorkItemsPerXForm, dataFormat)

	elif N >= mem_coalesce_width:
		numInnerIter = N / mem_coalesce_width
		numOuterIter = numXFormsPerWG / ( groupSize / mem_coalesce_width )

		res += "	ii = lId & " + str(mem_coalesce_width - 1) + ";\n"
		res += "	jj = lId >> " + str(log2(mem_coalesce_width)) + ";\n"
		res += "	lMemStore = mad24( jj, " + str(N + numWorkItemsPerXForm) + ", ii );\n"
		res += "	offset = mad24( groupId, " + str(numXFormsPerWG) + ", jj);\n"
		res += "	offset = mad24( offset, " + str(N) + ", ii );\n"

		if dataFormat == clFFT_InterleavedComplexFormat:
			res += "		in += offset;\n"
			res += "		out += offset;\n"
		else:
			res += "		in_real += offset;\n"
			res += "		in_imag += offset;\n"
			res += "		out_real += offset;\n"
			res += "		out_imag += offset;\n"

		res += "if((groupId == gridDim.x - 1) && s) {\n"
		for i in range(numOuterIter):
			res += "	if( jj < s ) {\n"
			for j in range(numInnerIter):
				res += formattedLoad(i * numInnerIter + j,
					j * mem_coalesce_width + i * ( groupSize / mem_coalesce_width ) * N,
					dataFormat)
			res += "	}\n"
			if i != numOuterIter - 1:
				res += "	jj += " + str(groupSize / mem_coalesce_width) + ";\n"

		res += "}\n "
		res += "else {\n"
		for i in range(numOuterIter):
			for j in range(numInnerIter):
				res += formattedLoad(i * numInnerIter + j,
					j * mem_coalesce_width + i * ( groupSize / mem_coalesce_width ) * N,
					dataFormat)

		res += "}\n"

		res += "	ii = lId & " + str(numWorkItemsPerXForm - 1) + ";\n"
		res += "	jj = lId >> " + str(log2NumWorkItemsPerXForm) + ";\n"
		res += "	lMemLoad  = mad24( jj, " + str(N + numWorkItemsPerXForm) + ", ii);\n"

		for i in range(numOuterIter):
			for j in range(numInnerIter):
				res += "	sMem[lMemStore+" + str(j * mem_coalesce_width +
					i * ( groupSize / mem_coalesce_width ) * (N + numWorkItemsPerXForm )) + "] = a[" + \
					str(i * numInnerIter + j) + "].x;\n"

		res += "	__syncthreads();\n"

		for i in range(R0):
			res += "	a[" + str(i) + "].x = sMem[lMemLoad+" + str(i * numWorkItemsPerXForm) + "];\n"
		res += "	__syncthreads();\n"

		for i in range(numOuterIter):
			for j in range(numInnerIter):
				res += "	sMem[lMemStore+" + str(j * mem_coalesce_width +
					i * ( groupSize / mem_coalesce_width ) * (N + numWorkItemsPerXForm )) + "] = a[" + \
					str(i * numInnerIter + j) + "].y;\n"

		res += "	__syncthreads();\n"

		for i in range(R0):
			res += "	a[" + str(i) + "].y = sMem[lMemLoad+" + str(i * numWorkItemsPerXForm) + "];\n"
		res += "	__syncthreads();\n"

		lMemSize = (N + numWorkItemsPerXForm) * numXFormsPerWG

	else:
		res += "	offset = mad24( groupId,  " + str(N * numXFormsPerWG) + ", lId );\n"
		if dataFormat == clFFT_InterleavedComplexFormat:
			res += "		in += offset;\n"
			res += "		out += offset;\n"
		else:
			res += "		in_real += offset;\n"
			res += "		in_imag += offset;\n"
			res += "		out_real += offset;\n"
			res += "		out_imag += offset;\n"

		res += "	ii = lId & " + str(N-1) + ";\n"
		res += "	jj = lId >> " + str(log2(N)) + ";\n"
		res += "	lMemStore = mad24( jj, " + str(N + numWorkItemsPerXForm) + ", ii );\n"

		res += "if((groupId == gridDim.x - 1) && s) {\n"
		for i in range(R0):
			res += "	if(jj < s )\n"
			res += formattedLoad(i, i*groupSize, dataFormat)
			if i != R0 - 1:
				res += "	jj += " + str(groupSize / N) + ";\n"

		res += "}\n"

		res += "else {\n"

		for i in range(R0):
			res += formattedLoad(i, i*groupSize, dataFormat)

		res += "}\n"

		if numWorkItemsPerXForm > 1:
			res += "	ii = lId & " + str(numWorkItemsPerXForm - 1) + ";\n"
			res += "	jj = lId >> " + str(log2NumWorkItemsPerXForm) + ";\n"
			res += "	lMemLoad = mad24( jj, " + str(N + numWorkItemsPerXForm) + ", ii );\n"
		else:
			res += "	ii = 0;\n"
			res += "	jj = lId;\n"
			res += "	lMemLoad = mul24( jj, " + str(N + numWorkItemsPerXForm) + ");\n"

		for i in range(R0):
			res += "	sMem[lMemStore+" + str(i * ( groupSize / N ) * ( N + numWorkItemsPerXForm )) + "] = a[" + str(i) + "].x;\n"
		res += "	__syncthreads();\n"

		for i in range(R0):
			res += "	a[" + str(i) + "].x = sMem[lMemLoad+" + str(i * numWorkItemsPerXForm) + "];\n"
		res += "	__syncthreads();\n"

		for i in range(R0):
			res += "	sMem[lMemStore+" + str(i * ( groupSize / N ) * ( N + numWorkItemsPerXForm )) + "] = a[" + str(i) + "].y;\n"
		res += "	__syncthreads();\n"

		for i in range(R0):
			res += "	a[" + str(i) + "].y = sMem[lMemLoad+" + str(i * numWorkItemsPerXForm) + "];\n"
		res += "	__syncthreads();\n"

		lMemSize = (N + numWorkItemsPerXForm) * numXFormsPerWG

	return res, lMemSize

def insertGlobalStoresAndTranspose(N, maxRadix, Nr, numWorkItemsPerXForm, numXFormsPerWG, mem_coalesce_width, dataFormat):

	groupSize = numWorkItemsPerXForm * numXFormsPerWG
	lMemSize = 0
	numIter = maxRadix / Nr
	indent = ""
	res = ""

	if numWorkItemsPerXForm >= mem_coalesce_width:
		if numXFormsPerWG > 1:
			res += "	if( !s || (groupId < gridDim.x - 1) || (jj < s) ) {\n"
			indent = "	"

		for i in range(maxRadix):
			j = i % numIter
			k = i / numIter
			ind = j * Nr + k
			res += formattedStore(ind, i*numWorkItemsPerXForm, dataFormat)

		if numXFormsPerWG > 1:
			res += "	}\n"

	elif N >= mem_coalesce_width:
		numInnerIter = N / mem_coalesce_width
		numOuterIter = numXFormsPerWG / ( groupSize / mem_coalesce_width )

		res += "	lMemLoad  = mad24( jj, " + str(N + numWorkItemsPerXForm) + ", ii );\n"
		res += "	ii = lId & " + str(mem_coalesce_width - 1) + ";\n"
		res += "	jj = lId >> " + str(log2(mem_coalesce_width)) + ";\n"
		res += "	lMemStore = mad24( jj," + str(N + numWorkItemsPerXForm) + ", ii );\n"

		for i in range(maxRadix):
			j = i % numIter
			k = i / numIter
			ind = j * Nr + k
			res += "	sMem[lMemLoad+" + str(i*numWorkItemsPerXForm) + "] = a[" + str(ind) + "].x;\n"

		res += "	__syncthreads();\n"

		for i in range(numOuterIter):
			for j in range(numInnerIter):
				res += "	a[" + str(i*numInnerIter + j) + \
					"].x = sMem[lMemStore+" + str(j*mem_coalesce_width +
					i*( groupSize / mem_coalesce_width )*(N + numWorkItemsPerXForm)) + "];\n"
		res += "	__syncthreads();\n"

		for i in range(maxRadix):
			j = i % numIter
			k = i / numIter
			ind = j * Nr + k
			res += "	sMem[lMemLoad+" + str(i*numWorkItemsPerXForm) + "] = a[" + str(ind) + "].y;\n"

		res += "	__syncthreads();\n"

		for i in range(numOuterIter):
			for j in range(numInnerIter):
				res += "	a[" + str(i*numInnerIter + j) + "].y = sMem[lMemStore+" + \
					str(j*mem_coalesce_width + i*( groupSize / mem_coalesce_width )*(N + numWorkItemsPerXForm)) + "];\n"
		res += "	__syncthreads();\n"

		res += "if((groupId == gridDim.x - 1) && s) {\n"
		for i in range(numOuterIter):
			res += "	if( jj < s ) {\n"
			for j in range(numInnerIter):
				res += formattedStore(i*numInnerIter + j,
					j*mem_coalesce_width + i*(groupSize/mem_coalesce_width)*N,
					dataFormat)
			res += "	}\n"
			if i != numOuterIter - 1:
				res += "	jj += " + str(groupSize / mem_coalesce_width) + ";\n"

		res += "}\n"
		res += "else {\n"
		for i in range(numOuterIter):
			for j in range(numInnerIter):
				res += formattedStore(i*numInnerIter + j,
					j*mem_coalesce_width + i*(groupSize/mem_coalesce_width)*N,
					dataFormat)

		res += "}\n"

		lMemSize = (N + numWorkItemsPerXForm) * numXFormsPerWG

	else:
		res += "	lMemLoad  = mad24( jj," + str(N + numWorkItemsPerXForm) + ", ii );\n"

		res += "	ii = lId & " + str(N - 1) + ";\n"
		res += "	jj = lId >> " + str(log2(N)) + ";\n"
		res += "	lMemStore = mad24( jj," + str(N + numWorkItemsPerXForm) + ", ii );\n"

		for i in range(maxRadix):
			j = i % numIter
			k = i / numIter
			ind = j * Nr + k
			res += "	sMem[lMemLoad+" + str(i*numWorkItemsPerXForm) + "] = a[" + str(ind) + "].x;\n"

		res += "	__syncthreads();\n"

		for i in range(maxRadix):
			res += "	a[" + str(i) + "].x = sMem[lMemStore+" + str(i*( groupSize / N )*( N + numWorkItemsPerXForm )) + "];\n"
		res += "	__syncthreads();\n"

		for i in range(maxRadix):
			j = i % numIter
			k = i / numIter
			ind = j * Nr + k
			res += "	sMem[lMemLoad+" + str(i*numWorkItemsPerXForm) + "] = a[" + str(ind) + "].y;\n"

		res += "	__syncthreads();\n"

		for i in range(maxRadix):
			res += "	a[" + str(i) + "].y = sMem[lMemStore+" + str(i*( groupSize / N )*( N + numWorkItemsPerXForm )) + "];\n"
		res += "	__syncthreads();\n"
		# XXX
		res += "if((groupId == gridDim.x - 1) && s) {\n"
		for i in range(maxRadix):
			res += "	if(jj < s ) {\n"
			res += formattedStore(i, i*groupSize, dataFormat)
			res += "	}\n"
			if i != maxRadix - 1:
				res += "	jj +=" + str(groupSize / N) + ";\n"

		res += "}\n"
		res += "else {\n"
		for i in range(maxRadix):
			res += formattedStore(i, i*groupSize, dataFormat)

		res += "}\n"

		lMemSize = (N + numWorkItemsPerXForm) * numXFormsPerWG

	return res, lMemSize

def insertfftKernel(Nr, numIter):
	#return "" # XXX
	res = ""
	for i in range(numIter):
		res += "	fftKernel" + str(Nr) + "(a+" + str(i*Nr) + ", dir);\n"
	return res

def insertTwiddleKernel(Nr, numIter, Nprev, len, numWorkItemsPerXForm):

	logNPrev = log2(Nprev)
	res = ""

	for z in range(numIter):
		if z == 0:
			if Nprev > 1:
				res += "	angf = (float) (ii >> " + str(logNPrev) + ");\n"
			else:
				res += "	angf = (float) ii;\n"
		else:
			if Nprev > 1:
				res += "	angf = (float) ((" + str(z*numWorkItemsPerXForm) + " + ii) >>" + str(logNPrev) + ");\n"
			else:
				# TODO: find out which conditions are necessary to execute this code
				res += "	angf = (float) (" + str(z*numWorkItemsPerXForm) + " + ii);\n"

		for k in range(1, Nr):
			ind = z*Nr + k
			res += "	ang = dir * ( 2.0f * M_PI * " + str(k) + ".0f / " + str(len) + ".0f )" + " * angf;\n"
			# TODO: use native_cos and sin (as OpenCL implementation did)
			res += "	w = make_float2(cos(ang), sin(ang));\n"
			res += "	a[" + str(ind) + "] = complexMul(a[" + str(ind) + "], w);\n"
	return res

def getPadding(numWorkItemsPerXForm, Nprev, numWorkItemsReq, numXFormsPerWG, Nr, numBanks):

	if numWorkItemsPerXForm <= Nprev or Nprev >= numBanks:
		offset = 0
	else:
		numRowsReq = (numWorkItemsPerXForm if numWorkItemsPerXForm < numBanks else numBanks) / Nprev
		numColsReq = 1
		if numRowsReq > Nr:
			numColsReq = numRowsReq / Nr
		numColsReq = Nprev * numColsReq
		offset = numColsReq

	if numWorkItemsPerXForm >= numBanks or numXFormsPerWG == 1:
		midPad = 0
	else:
		bankNum = ( (numWorkItemsReq + offset) * Nr ) & (numBanks - 1)
		if bankNum >= numWorkItemsPerXForm:
			midPad = 0
		else:
			# TODO: find out which conditions are necessary to execute this code
			midPad = numWorkItemsPerXForm - bankNum

	lMemSize = ( numWorkItemsReq + offset) * Nr * numXFormsPerWG + midPad * (numXFormsPerWG - 1)
	return lMemSize, offset, midPad

def insertLocalStores(numIter, Nr, numWorkItemsPerXForm, numWorkItemsReq, offset, comp):
	res = ""
	for z in range(numIter):
		for k in range(Nr):
			index = k*(numWorkItemsReq + offset) + z*numWorkItemsPerXForm
			res += "	sMem[lMemStore+" + str(index) + "] = a[" + str(z*Nr + k) + "]." + comp + ";\n"

	res += "	__syncthreads();\n"
	return res

def insertLocalLoads(n, Nr, Nrn, Nprev, Ncurr, numWorkItemsPerXForm, numWorkItemsReq, offset, comp):
	numWorkItemsReqN = n / Nrn
	interBlockHNum = max( Nprev / numWorkItemsPerXForm, 1 )
	interBlockHStride = numWorkItemsPerXForm
	vertWidth = max(numWorkItemsPerXForm / Nprev, 1)
	vertWidth = min( vertWidth, Nr)
	vertNum = Nr / vertWidth
	vertStride = ( n / Nr + offset ) * vertWidth
	iter = max( numWorkItemsReqN / numWorkItemsPerXForm, 1)
	intraBlockHStride = (numWorkItemsPerXForm / (Nprev*Nr)) if (numWorkItemsPerXForm / (Nprev*Nr)) > 1 else 1
	intraBlockHStride *= Nprev

	stride = numWorkItemsReq / Nrn

	res = ""
	for i in range(iter):
		ii = i / (interBlockHNum * vertNum)
		zz = i % (interBlockHNum * vertNum)
		jj = zz % interBlockHNum
		kk = zz / interBlockHNum

		for z in range(Nrn):
			st = kk * vertStride + jj * interBlockHStride + ii * intraBlockHStride + z * stride
			res += "	a[" + str(i*Nrn + z) + "]." + comp + " = sMem[lMemLoad+" + str(st) + "];\n"

	res += "	__syncthreads();\n"
	return res

def insertLocalLoadIndexArithmatic(Nprev, Nr, numWorkItemsReq, numWorkItemsPerXForm, numXFormsPerWG, offset, midPad):
	res = ""
	Ncurr = Nprev * Nr
	logNcurr = log2(Ncurr)
	logNprev = log2(Nprev)
	incr = (numWorkItemsReq + offset) * Nr + midPad

	if Ncurr < numWorkItemsPerXForm:
		if Nprev == 1:
			res += "	j = ii & " + str(Ncurr - 1) + ";\n"
		else:
			res += "	j = (ii & " + str(Ncurr - 1) + ") >> " + str(logNprev) + ";\n"

		if Nprev == 1:
			res += "	i = ii >> " + str(logNcurr) + ";\n"
		else:
			res += "	i = mad24(ii >> " + str(logNcurr) + ", " + str(Nprev) + ", ii & " + str(Nprev - 1) + ");\n"
	else:
		if Nprev == 1:
			res += "	j = ii;\n"
		else:
			res += "	j = ii >> " + str(logNprev) + ";\n"
		if Nprev == 1:
			res += "	i = 0;\n"
		else:
			res += "	i = ii & " + str(Nprev - 1) + ";\n"

	if numXFormsPerWG > 1:
		res += "	i = mad24(jj, " + str(incr) + ", i);\n"

	res += "	lMemLoad = mad24(j, " + str(numWorkItemsReq + offset) + ", i);\n"
	return res

def insertLocalStoreIndexArithmatic(numWorkItemsReq, numXFormsPerWG, Nr, offset, midPad):
	if numXFormsPerWG == 1:
		return "	lMemStore = ii;\n"
	else:
		return "	lMemStore = mad24(jj, " + str((numWorkItemsReq + offset)*Nr + midPad) + ", ii);\n"


def createLocalMemfftKernelString(plan, dataFormat):

	n = plan.n.x
	assert n <= plan.max_work_item_per_workgroup * plan.max_radix, "signal lenght too big for local mem fft"

	radixArray = getRadixArray(n, 0)
	assert len(radixArray) > 0, "no radix array supplied"

	if n / radixArray[0] > plan.max_work_item_per_workgroup:
		radixArray = getRadixArray(n, plan.max_radix)

	assert radixArray[0] <= plan.max_radix, "max radix choosen is greater than allowed"
	assert n / radixArray[0] <= plan.max_work_item_per_workgroup, \
		"required work items per xform greater than maximum work items allowed per work group for local mem fft"

	tmpLen = 1
	for i in range(len(radixArray)):
		# TODO: need some log2 check here
		# assert radixArray[i] && !((radixArray[i] - 1) & radixArray[i] )
		tmpLen *= radixArray[i]

	assert tmpLen == n, "product of radices choosen doesnt match the length of signal"

	localString = ""

	kCount = len(plan.kernel_info)
	kernelName = "fft" + str(kCount);

	kInfo = cl_fft_kernel_info()
	kInfo.kernel = 0
	kInfo.lmem_size = 0
	kInfo.num_workgroups = 0
	kInfo.num_workitems_per_workgroup = 0
	kInfo.dir = cl_fft_kernel_x
	kInfo.in_place_possible = 1
	kInfo.kernel_name = kernelName
	plan.kernel_info.append(kInfo)

	numWorkItemsPerXForm = n / radixArray[0]
	numWorkItemsPerWG = 64 if numWorkItemsPerXForm <= 64 else numWorkItemsPerXForm
	assert numWorkItemsPerWG <= plan.max_work_item_per_workgroup
	numXFormsPerWG = numWorkItemsPerWG / numWorkItemsPerXForm
	kInfo.num_workgroups = numXFormsPerWG
	kInfo.num_workitems_per_workgroup = numWorkItemsPerWG

	N = radixArray
	maxRadix = N[0]
	lMemSize = 0

	localString += insertVariables(maxRadix)

	res, lMemSize = insertGlobalLoadsAndTranspose(n, numWorkItemsPerXForm, numXFormsPerWG, maxRadix,
		plan.min_mem_coalesce_width, dataFormat)
	localString += res

	kInfo.lmem_size =  lMemSize if lMemSize > kInfo.lmem_size else kInfo.lmem_size

	xcomp = "x"
	ycomp = "y"

	Nprev = 1
	len_ = n

	numRadix = len(radixArray)
	for r in range(numRadix):

		numIter = N[0] / N[r]
		numWorkItemsReq = n / N[r]
		Ncurr = Nprev * N[r]
		localString += insertfftKernel(N[r], numIter)

		if r < numRadix - 1:
			localString += insertTwiddleKernel(N[r], numIter, Nprev, len_, numWorkItemsPerXForm);
			lMemSize, offset, midPad = getPadding(numWorkItemsPerXForm, Nprev, numWorkItemsReq, numXFormsPerWG, N[r], plan.num_local_mem_banks)
			kInfo.lmem_size = lMemSize if lMemSize > kInfo.lmem_size else kInfo.lmem_size
			localString += insertLocalStoreIndexArithmatic(numWorkItemsReq, numXFormsPerWG, N[r], offset, midPad)
			localString += insertLocalLoadIndexArithmatic(Nprev, N[r], numWorkItemsReq, numWorkItemsPerXForm, numXFormsPerWG, offset, midPad)
			localString += insertLocalStores(numIter, N[r], numWorkItemsPerXForm, numWorkItemsReq, offset, xcomp)
			localString += insertLocalLoads(n, N[r], N[r+1], Nprev, Ncurr, numWorkItemsPerXForm, numWorkItemsReq, offset, xcomp)
			localString += insertLocalStores(numIter, N[r], numWorkItemsPerXForm, numWorkItemsReq, offset, ycomp)
			localString += insertLocalLoads(n, N[r], N[r+1], Nprev, Ncurr, numWorkItemsPerXForm, numWorkItemsReq, offset, ycomp)
			Nprev = Ncurr
			len_ = len_ / N[r]

	res, lMemSize = insertGlobalStoresAndTranspose(n, maxRadix, N[numRadix - 1], numWorkItemsPerXForm, numXFormsPerWG, plan.min_mem_coalesce_width, dataFormat)
	localString += res

	kInfo.lmem_size = lMemSize if lMemSize > kInfo.lmem_size else kInfo.lmem_size

	if kInfo.lmem_size > 0:
		localString = "	__shared__ float sMem[" + str(kInfo.lmem_size) + "];\n" + localString

	localString = insertHeader(kernelName, dataFormat) + "{\n" + localString
	localString += "}\n"

	return localString

# For n larger than what can be computed using local memory fft, global transposes
# multiple kernel launces is needed. For these sizes, n can be decomposed using
# much larger base radices i.e. say n = 262144 = 128 x 64 x 32. Thus three kernel
# launches will be needed, first computing 64 x 32, length 128 ffts, second computing
# 128 x 32 length 64 ffts, and finally a kernel computing 128 x 64 length 32 ffts.
# Each of these base radices can futher be divided into factors so that each of these
# base ffts can be computed within one kernel launch using in-register ffts and local
# memory transposes i.e for the first kernel above which computes 64 x 32 ffts on length
# 128, 128 can be decomposed into 128 = 16 x 8 i.e. 8 work items can compute 8 length
# 16 ffts followed by transpose using local memory followed by each of these eight
# work items computing 2 length 8 ffts thus computing 16 length 8 ffts in total. This
# means only 8 work items are needed for computing one length 128 fft. If we choose
# work group size of say 64, we can compute 64/8 = 8 length 128 ffts within one
# work group. Since we need to compute 64 x 32 length 128 ffts in first kernel, this
# means we need to launch 64 x 32 / 8 = 256 work groups with 64 work items in each
# work group where each work group is computing 8 length 128 ffts where each length
# 128 fft is computed by 8 work items. Same logic can be applied to other two kernels
# in this example. Users can play with difference base radices and difference
# decompositions of base radices to generates different kernels and see which gives
# best performance. Following function is just fixed to use 128 as base radix

def getGlobalRadixInfo(n):
	baseRadix = min(n, 128)

	numR = 0
	N = n
	while N > baseRadix:
		N /= baseRadix
		numR += 1

	radix = []
	for i in range(numR):
		radix.append(baseRadix)

	radix.append(N)
	numR += 1

	R1 = []
	R2 = []
	for i in range(numR):
		B = radix[i]
		if B <= 8:
			R1.append(B)
			R2.append(1)
		else:
			r1 = 2
			r2 = B / r1
			while r2 > r1:
			   r1 *= 2
			   r2 = B / r1

			R1.append(r1)
			R2.append(r2)

	return radix, R1, R2

def createGlobalFFTKernelString(plan, n, BS, dir, vertBS, dataFormat):

	maxThreadsPerBlock = plan.max_work_item_per_workgroup
	maxArrayLen = plan.max_radix
	batchSize = plan.min_mem_coalesce_width
	vertical = 0 if dir == cl_fft_kernel_x else 1

	radixArr, R1Arr, R2Arr = getGlobalRadixInfo(n)

	numPasses = len(radixArr)

	localString = ""
	kCount = len(plan.kernel_info)

	N = n
	m = log2(n)
	Rinit = BS if vertical else 1
	batchSize = min(BS, batchSize) if vertical else batchSize

	localString = ""

	for passNum in range(numPasses):

		kernelName = ""

		radix = radixArr[passNum]
		R1 = R1Arr[passNum]
		R2 = R2Arr[passNum]

		strideI = Rinit
		for i in range(numPasses):
			if i != passNum:
				strideI *= radixArr[i]

		strideO = Rinit
		for i in range(passNum):
			strideO *= radixArr[i]

		threadsPerXForm = R2
		batchSize = plan.max_work_item_per_workgroup if R2 == 1 else batchSize
		batchSize = min(batchSize, strideI)
		threadsPerBlock = batchSize * threadsPerXForm
		threadsPerBlock = min(threadsPerBlock, maxThreadsPerBlock)
		batchSize = threadsPerBlock / threadsPerXForm
		assert R2 <= R1
		assert R1*R2 == radix
		assert R1 <= maxArrayLen
		assert threadsPerBlock <= maxThreadsPerBlock

		numIter = R1 / R2
		gInInc = threadsPerBlock / batchSize

		lgStrideO = log2(strideO)
		numBlocksPerXForm = strideI / batchSize
		numBlocks = numBlocksPerXForm
		if not vertical:
			numBlocks *= BS
		else:
			numBlocks *= vertBS

		kernelName = "fft" + str(kCount)
		kCount += 1
		kInfo = cl_fft_kernel_info()
		kInfo.kernel = 0
		if R2 == 1:
			kInfo.lmem_size = 0
		else:
			if strideO == 1:
				kInfo.lmem_size = (radix + 1)*batchSize
			else:
				kInfo.lmem_size = threadsPerBlock*R1

		kInfo.num_workgroups = numBlocks
		kInfo.num_workitems_per_workgroup = threadsPerBlock
		kInfo.dir = dir
		if passNum == numPasses - 1 and numPasses % 2 == 1:
			kInfo.in_place_possible = True
		else:
			kInfo.in_place_possible = False

		kInfo.kernel_name = kernelName
		plan.kernel_info.append(kInfo)

		localString += insertHeader(kernelName, dataFormat)
		localString += "{\n"
		if kInfo.lmem_size > 0:
			localString += "	__shared__ float sMem[" + str(kInfo.lmem_size) + "];\n"

		localString += insertVariables(R1)

		if vertical:
			localString += "xNum = groupId >> " + str(log2(numBlocksPerXForm)) + ";\n"
			localString += "groupId = groupId & " + str(numBlocksPerXForm - 1) + ";\n"
			localString += "indexIn = mad24(groupId, " + str(batchSize) + ", xNum << " + str(log2(n*BS)) + ");\n"
			localString += "tid = mul24(groupId, " + str(batchSize) + ");\n"
			localString += "i = tid >> " + str(lgStrideO) + ";\n"
			localString += "j = tid & " + str(strideO - 1) + ";\n"
			stride = radix*Rinit
			for i in range(passNum):
				stride *= radixArr[i]
			localString += "indexOut = mad24(i, " + str(stride) + ", j + " + "(xNum << " + str(log2(n*BS)) + "));\n"
			localString += "bNum = groupId;\n"
		else:
			lgNumBlocksPerXForm = log2(numBlocksPerXForm)
			localString += "bNum = groupId & " + str(numBlocksPerXForm - 1) + ";\n"
			localString += "xNum = groupId >> " + str(lgNumBlocksPerXForm) + ";\n"
			localString += "indexIn = mul24(bNum, " + str(batchSize) + ");\n"
			localString += "tid = indexIn;\n"
			localString += "i = tid >> " + str(lgStrideO) + ";\n"
			localString += "j = tid & " + str(strideO - 1) + ";\n"
			stride = radix*Rinit
			for i in range(passNum):
				stride *= radixArr[i]
			localString += "indexOut = mad24(i, " + str(stride) + ", j);\n"
			localString += "indexIn += (xNum << " + str(m) + ");\n"
			localString += "indexOut += (xNum << " + str(m) + ");\n"

		# Load Data
		lgBatchSize = log2(batchSize)
		localString += "tid = lId;\n"
		localString += "i = tid & " + str(batchSize - 1) + ";\n"
		localString += "j = tid >> " + str(lgBatchSize) + ";\n"
		localString += "indexIn += mad24(j, " + str(strideI) + ", i);\n"

		if dataFormat == clFFT_SplitComplexFormat:
			localString += "in_real += indexIn;\n"
			localString += "in_imag += indexIn;\n"
			for j in range(R1):
				localString += "a[" + str(j) + "].x = in_real[" + str(j*gInInc*strideI) + "];\n"
			for j in range(R1):
				localString += "a[" + str(j) + "].y = in_imag[" + str(j*gInInc*strideI) + "];\n"
		else:
			localString += "in += indexIn;\n"
			for j in range(R1):
				localString += "a[" + str(j) + "] = in[" + str(j*gInInc*strideI) + "];\n"

		localString += "fftKernel" + str(R1) + "(a, dir);\n"

		if R2 > 1:
			# twiddle
			for k in range(1, R1):
				localString += "ang = dir*(2.0f*M_PI*" + str(k) + "/" + str(radix) + ")*j;\n"
				# TODO: use native cos and sin (as OpenCL implementation did)
				localString += "w = make_float2(cos(ang), sin(ang));\n"
				localString += "a[" + str(k) + "] = complexMul(a[" + str(k) + "], w);\n"

			# shuffle
			numIter = R1 / R2
			localString += "indexIn = mad24(j, " + str(threadsPerBlock*numIter) + ", i);\n"
			localString += "lMemStore = tid;\n"
			localString += "lMemLoad = indexIn;\n"
			for k in range(R1):
				localString += "sMem[lMemStore+" + str(k*threadsPerBlock) + "] = a[" + str(k) + "].x;\n"
			localString += "__syncthreads();\n"
			for k in range(numIter):
				for t in range(R2):
					localString += "a[" + str(k*R2+t) + "].x = sMem[lMemLoad+" + str(t*batchSize + k*threadsPerBlock) + "];\n"
			localString += "__syncthreads();\n"
			for k in range(R1):
				localString += "sMem[lMemStore+" + str(k*threadsPerBlock) + "] = a[" + str(k) + "].y;\n"
			localString += "__syncthreads();\n"
			for k in range(numIter):
				for t in range(R2):
					localString += "a[" + str(k*R2+t) + "].y = sMem[lMemLoad+" + str(t*batchSize + k*threadsPerBlock) + "];\n"
			localString += "__syncthreads();\n"

			for j in range(numIter):
				localString += "fftKernel" + str(R2) + "(a + " + str(j*R2) + ", dir);\n"

		# twiddle
		if passNum < numPasses - 1:
			localString += "l = ((bNum << " + str(lgBatchSize) + ") + i) >> " + str(lgStrideO) + ";\n"
			localString += "k = j << " + str(log2(R1/R2)) + ";\n"
			localString += "ang1 = dir*(2.0f*M_PI/" + str(N) + ")*l;\n"
			for t in range(R1):
				localString += "ang = ang1*(k + " + str((t%R2)*R1 + (t/R2)) + ");\n"
				# TODO: use native cos and sin (as OpenCL implementation did)
				localString += "w = make_float2(cos(ang), sin(ang));\n"
				localString += "a[" + str(t) + "] = complexMul(a[" + str(t) + "], w);\n"

		# Store Data
		if strideO == 1:
			localString += "lMemStore = mad24(i, " + str(radix + 1) + ", j << " + str(log2(R1/R2)) + ");\n"
			localString += "lMemLoad = mad24(tid >> " + str(log2(radix)) + \
				", " + str(radix+1) + ", tid & " + str(radix-1) + ");\n"

			for i in range(R1/R2):
				for j in range(R2):
					localString += "sMem[lMemStore+" + str(i + j*R1) + "] = a[" + str(i*R2+j) + "].x;\n"
			localString += "__syncthreads();\n"
			for i in range(R1):
				localString += "a[" + str(i) + "].x = sMem[lMemLoad+" + str(i*(radix+1)*(threadsPerBlock/radix)) + "];\n"
			localString += "__syncthreads();\n"

			for i in range(R1/R2):
				for j in range(R2):
					localString += "sMem[lMemStore+" + str(i + j*R1) + "] = a[" + str(i*R2+j) + "].y;\n"
			localString += "__syncthreads();\n"
			for i in range(R1):
				localString += "a[" + str(i) + "].y = sMem[lMemLoad+" + str(i*(radix+1)*(threadsPerBlock/radix)) + "];\n"
			localString += "__syncthreads();\n"

			localString += "indexOut += tid;\n"
			if dataFormat == clFFT_SplitComplexFormat:
				localString += "out_real += indexOut;\n"
				localString += "out_imag += indexOut;\n"
				for k in range(R1):
					localString += "out_real[" + str(k*threadsPerBlock) + "] = a[" + str(k) + "].x;\n"
				for k in range(R1):
					localString += "out_imag[" + str(k*threadsPerBlock) + "] = a[" + str(k) + "].y;\n"
			else:
				localString += "out += indexOut;\n"
				for k in range(R1):
					localString += "out[" + str(k*threadsPerBlock) + "] = a[" + str(k) + "];\n"
		else:
			localString += "indexOut += mad24(j, " + str(numIter*strideO) + ", i);\n"
			if dataFormat == clFFT_SplitComplexFormat:
				localString += "out_real += indexOut;\n"
				localString += "out_imag += indexOut;\n"
				for k in range(R1):
					localString += "out_real[" + str(((k%R2)*R1 + (k/R2))*strideO) + "] = a[" + str(k) + "].x;\n"
				for k in range(R1):
					localString += "out_imag[" + str(((k%R2)*R1 + (k/R2))*strideO) + "] = a[" + str(k) + "].y;\n"
			else:
				localString += "out += indexOut;\n"
				for k in range(R1):
					localString += "out[" + str(((k%R2)*R1 + (k/R2))*strideO) + "] = a[" + str(k) + "];\n"

		localString += "}\n"

		N /= radix

	return localString

def FFT1D(plan, dir):

	if dir == cl_fft_kernel_x:
		if plan.n.x > plan.max_localmem_fft_size:
			return createGlobalFFTKernelString(plan, plan.n.x, 1, cl_fft_kernel_x, 1, plan.dataFormat)
		elif plan.n.x > 1:
			radixArray = getRadixArray(plan.n.x, 0)
			if plan.n.x / radixArray[0] <= plan.max_work_item_per_workgroup:
				return createLocalMemfftKernelString(plan, plan.dataFormat);
			else:
				radixArray = getRadixArray(plan.n.x, plan.max_radix)
				if plan.n.x / radixArray[0] <= plan.max_work_item_per_workgroup:
					return createLocalMemfftKernelString(plan, plan.dataFormat)
				else:
					# TODO: find out which conditions are necessary to execute this code
					return createGlobalFFTKernelString(plan, plan.n.x, 1, cl_fft_kernel_x, 1, plan.dataFormat)
		else:
			return ""
	elif dir == cl_fft_kernel_y:
		if plan.n.y > 1:
			return createGlobalFFTKernelString(plan, plan.n.y, plan.n.x, cl_fft_kernel_y, 1, plan.dataFormat)
		else:
			return ""
	elif dir == cl_fft_kernel_z:
		if plan.n.z > 1:
			return createGlobalFFTKernelString(plan, plan.n.z, plan.n.x*plan.n.y, cl_fft_kernel_z, 1, plan.dataFormat)
		else:
			return ""
	else:
		raise Exception("Wrong direction")
