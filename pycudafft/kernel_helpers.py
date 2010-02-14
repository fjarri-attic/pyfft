# TODO: think of something more effective
def log2(n):
	pos = 0
	for pow in [16, 8, 4, 2, 1]:
		if n >= 2 ** pow:
			n /= (2 ** pow)
			pos += pow
	return pos

def getRadixArray(n, max_radix):
	"""
	For any n, this function decomposes n into factors for loacal memory tranpose
	based fft. Factors (radices) are sorted such that the first one (radix_array[0])
	is the largest. This base radix determines the number of registers used by each
	work item and product of remaining radices determine the size of work group needed.
	To make things concrete with and example, suppose n = 1024. It is decomposed into
	1024 = 16 x 16 x 4. Hence kernel uses float2 a[16], for local in-register fft and
	needs 16 x 4 = 64 work items per work group. So kernel first performance 64 length
	16 ffts (64 work items working in parallel) following by transpose using local
	memory followed by again 64 length 16 ffts followed by transpose using local memory
	followed by 256 length 4 ffts. For the last step since with size of work group is
	64 and each work item can array for 16 values, 64 work items can compute 256 length
	4 ffts by each work item computing 4 length 4 ffts.
	Similarly for n = 2048 = 8 x 8 x 8 x 4, each work group has 8 x 8 x 4 = 256 work
	iterms which each computes 256 (in-parallel) length 8 ffts in-register, followed
	by transpose using local memory, followed by 256 length 8 in-register ffts, followed
	by transpose using local memory, followed by 256 length 8 in-register ffts, followed
	by transpose using local memory, followed by 512 length 4 in-register ffts. Again,
	for the last step, each work item computes two length 4 in-register ffts and thus
	256 work items are needed to compute all 512 ffts.
	For n = 32 = 8 x 4, 4 work items first compute 4 in-register
	lenth 8 ffts, followed by transpose using local memory followed by 8 in-register
	length 4 ffts, where each work item computes two length 4 ffts thus 4 work items
	can compute 8 length 4 ffts. However if work group size of say 64 is choosen,
	each work group can compute 64/ 4 = 16 size 32 ffts (batched transform).
	Users can play with these parameters to figure what gives best performance on
	their particular device i.e. some device have less register space thus using
	smaller base radix can avoid spilling ... some has small local memory thus
	using smaller work group size may be required etc
	"""
	if max_radix > 1:
		max_radix = min(n, max_radix)
		radix_array = []
		while n > max_radix:
			radix_array.append(max_radix)
			n /= max_radix
		radix_array.append(n)
		return radix_array

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

def getGlobalRadixInfo(n):
	"""
	For n larger than what can be computed using local memory fft, global transposes
	multiple kernel launces is needed. For these sizes, n can be decomposed using
	much larger base radices i.e. say n = 262144 = 128 x 64 x 32. Thus three kernel
	launches will be needed, first computing 64 x 32, length 128 ffts, second computing
	128 x 32 length 64 ffts, and finally a kernel computing 128 x 64 length 32 ffts.
	Each of these base radices can futher be divided into factors so that each of these
	base ffts can be computed within one kernel launch using in-register ffts and local
	memory transposes i.e for the first kernel above which computes 64 x 32 ffts on length
	128, 128 can be decomposed into 128 = 16 x 8 i.e. 8 work items can compute 8 length
	16 ffts followed by transpose using local memory followed by each of these eight
	work items computing 2 length 8 ffts thus computing 16 length 8 ffts in total. This
	means only 8 work items are needed for computing one length 128 fft. If we choose
	work group size of say 64, we can compute 64/8 = 8 length 128 ffts within one
	work group. Since we need to compute 64 x 32 length 128 ffts in first kernel, this
	means we need to launch 64 x 32 / 8 = 256 work groups with 64 work items in each
	work group where each work group is computing 8 length 128 ffts where each length
	128 fft is computed by 8 work items. Same logic can be applied to other two kernels
	in this example. Users can play with difference base radices and difference
	decompositions of base radices to generates different kernels and see which gives
	best performance. Following function is just fixed to use 128 as base radix
	"""
	base_radix = min(n, 128)

	numR = 0
	N = n
	while N > base_radix:
		N /= base_radix
		numR += 1

	radix = []
	for i in range(numR):
		radix.append(base_radix)

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

def getPadding(threads_per_xform, Nprev, threads_req, xforms_per_block, Nr, num_banks):

	if threads_per_xform <= Nprev or Nprev >= num_banks:
		offset = 0
	else:
		numRowsReq = (threads_per_xform if threads_per_xform < num_banks else num_banks) / Nprev
		numColsReq = 1
		if numRowsReq > Nr:
			numColsReq = numRowsReq / Nr
		numColsReq = Nprev * numColsReq
		offset = numColsReq

	if threads_per_xform >= num_banks or xforms_per_block == 1:
		midPad = 0
	else:
		bankNum = ((threads_req + offset) * Nr) & (num_banks - 1)
		if bankNum >= threads_per_xform:
			midPad = 0
		else:
			# TODO: find out which conditions are necessary to execute this code
			midPad = threads_per_xform - bankNum

	smem_size = (threads_req + offset) * Nr * xforms_per_block + midPad * (xforms_per_block - 1)
	return smem_size, offset, midPad

def getSharedMemorySize(n, radix_array, threads_per_xform, xforms_per_block, num_local_mem_banks, min_mem_coalesce_width):

	smem_size = 0

	# from insertGlobal(Loads/Stores)AndTranspose
	if threads_per_xform < min_mem_coalesce_width:
		smem_size = max(smem_size, (n + threads_per_xform) * xforms_per_block)

	Nprev = 1
	len_ = n
	numRadix = len(radix_array)
	for r in range(numRadix):

		numIter = radix_array[0] / radix_array[r]
		threads_req = n / radix_array[r]
		Ncurr = Nprev * radix_array[r]

		if r < numRadix - 1:
			smem_size_new, offset, midPad = getPadding(threads_per_xform, Nprev, threads_req, xforms_per_block,
				radix_array[r], num_local_mem_banks)
			smem_size = max(smem_size, smem_size_new)
			Nprev = Ncurr
			len_ = len_ / radix_array[r]

	return smem_size
