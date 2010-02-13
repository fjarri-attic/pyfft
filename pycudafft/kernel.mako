<%def name="baseKernels(scalar, complex)">

	#ifndef M_PI
	#define M_PI 0x1.921fb54442d18p+1
	#endif

	#define complex_ctr(x, y) make_${complex}(x, y)

	## TODO: replace by intrinsincs

	## multiplication + addition
	#define mad24(x, y, z) ((x) * (y) + (z))
	#define mad(x, y, z) ((x) * (y) + (z))

	## integer multiplication
	#define mul24(x, y) ((x) * (y))

	inline ${complex} operator+(${complex} a, ${complex} b) { return complex_ctr(a.x + b.x, a.y + b.y); }
	inline ${complex} operator-(${complex} a, ${complex} b) { return complex_ctr(a.x - b.x, a.y - b.y); }
	inline ${complex} operator*(${complex} a, ${scalar}  b) { return complex_ctr(b * a.x, b * a.y); }
	inline ${complex} operator*(${complex} a, ${complex} b) { return complex_ctr(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

	## This version (with mad() replaced by proper multiplication/addition function) would be
	## slower, but more precise (because CUDA compiler replaces ususal multiplications and
	## additions with non-IEEE compliant implementation)
	## Don't know whether this precision is necessary, so this will stay commented for a while
	##inline ${complex} operator *(${complex} a, ${complex} b) { \
	##	return complex_ctr(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y)); }

	#define conj(a) complex_ctr((a).x, -(a).y)
	#define conjTransp(a) complex_ctr(-(a).y, (a).x)

	#define fftKernel2(a, dir) \
	{ \
		${complex} c = (a)[0]; \
		(a)[0] = c + (a)[1]; \
		(a)[1] = c - (a)[1]; \
	}

	#define fftKernel2S(d1, d2, dir) \
	{ \
		${complex} c = (d1); \
		(d1) = c + (d2); \
		(d2) = c - (d2); \
	}

	#define fftKernel4(a, dir) \
	{ \
		fftKernel2S((a)[0], (a)[2], dir); \
		fftKernel2S((a)[1], (a)[3], dir); \
		fftKernel2S((a)[0], (a)[1], dir); \
		(a)[3] = conjTransp((a)[3]) * dir; \
		fftKernel2S((a)[2], (a)[3], dir); \
		${complex} c = (a)[1]; \
		(a)[1] = (a)[2]; \
		(a)[2] = c; \
	}

	#define fftKernel4s(a0, a1, a2, a3, dir) \
	{ \
		fftKernel2S((a0), (a2), dir); \
		fftKernel2S((a1), (a3), dir); \
		fftKernel2S((a0), (a1), dir); \
		(a3) = conjTransp((a3)) * dir; \
		fftKernel2S((a2), (a3), dir); \
		${complex} c = (a1); \
		(a1) = (a2); \
		(a2) = c; \
	}

	#define bitreverse8(a) \
	{ \
		${complex} c; \
		c = (a)[1]; \
		(a)[1] = (a)[4]; \
		(a)[4] = c; \
		c = (a)[3]; \
		(a)[3] = (a)[6]; \
		(a)[6] = c; \
	}

	#define fftKernel8(a, dir) \
	{ \
		const ${complex} w1  = complex_ctr(0x1.6a09e6p-1f,  dir*0x1.6a09e6p-1f); \
		const ${complex} w3  = complex_ctr(-0x1.6a09e6p-1f, dir*0x1.6a09e6p-1f); \
		fftKernel2S((a)[0], (a)[4], dir); \
		fftKernel2S((a)[1], (a)[5], dir); \
		fftKernel2S((a)[2], (a)[6], dir); \
		fftKernel2S((a)[3], (a)[7], dir); \
		(a)[5] = w1 * (a)[5]; \
		(a)[6] = conjTransp((a)[6]) * dir; \
		(a)[7] = w3 * (a)[7]; \
		fftKernel2S((a)[0], (a)[2], dir); \
		fftKernel2S((a)[1], (a)[3], dir); \
		fftKernel2S((a)[4], (a)[6], dir); \
		fftKernel2S((a)[5], (a)[7], dir); \
		(a)[3] = conjTransp((a)[3]) * dir; \
		(a)[7] = conjTransp((a)[7]) * dir; \
		fftKernel2S((a)[0], (a)[1], dir); \
		fftKernel2S((a)[2], (a)[3], dir); \
		fftKernel2S((a)[4], (a)[5], dir); \
		fftKernel2S((a)[6], (a)[7], dir); \
		bitreverse8((a)); \
	}

	#define bitreverse4x4(a) \
	{ \
		${complex} c; \
		c = (a)[1];  (a)[1]  = (a)[4];  (a)[4]  = c; \
		c = (a)[2];  (a)[2]  = (a)[8];  (a)[8]  = c; \
		c = (a)[3];  (a)[3]  = (a)[12]; (a)[12] = c; \
		c = (a)[6];  (a)[6]  = (a)[9];  (a)[9]  = c; \
		c = (a)[7];  (a)[7]  = (a)[13]; (a)[13] = c; \
		c = (a)[11]; (a)[11] = (a)[14]; (a)[14] = c; \
	}

	#define fftKernel16(a, dir) \
	{ \
		const ${scalar} w0 = 0x1.d906bcp-1f; \
		const ${scalar} w1 = 0x1.87de2ap-2f; \
		const ${scalar} w2 = 0x1.6a09e6p-1f; \
		fftKernel4s((a)[0], (a)[4], (a)[8],  (a)[12], dir); \
		fftKernel4s((a)[1], (a)[5], (a)[9],  (a)[13], dir); \
		fftKernel4s((a)[2], (a)[6], (a)[10], (a)[14], dir); \
		fftKernel4s((a)[3], (a)[7], (a)[11], (a)[15], dir); \
		(a)[5]  = (a)[5] * complex_ctr(w0, dir*w1); \
		(a)[6]  = (a)[6] * complex_ctr(w2, dir*w2); \
		(a)[7]  = (a)[7] * complex_ctr(w1, dir*w0); \
		(a)[9]  = (a)[9] * complex_ctr(w2, dir*w2); \
		(a)[10] = complex_ctr(dir, 0)*(conjTransp((a)[10])); \
		(a)[11] = (a)[11] * complex_ctr(-w2, dir*w2); \
		(a)[13] = (a)[13] * complex_ctr(w1, dir*w0); \
		(a)[14] = (a)[14] * complex_ctr(-w2, dir*w2); \
		(a)[15] = (a)[15] * complex_ctr(-w0, dir*-w1); \
		fftKernel4((a), dir); \
		fftKernel4((a) + 4, dir); \
		fftKernel4((a) + 8, dir); \
		fftKernel4((a) + 12, dir); \
		bitreverse4x4((a)); \
	}

	#define bitreverse32(a) \
	{ \
		${complex} c1, c2; \
		c1 = (a)[2];   (a)[2] = (a)[1];   c2 = (a)[4];   (a)[4] = c1;   c1 = (a)[8]; \
		(a)[8] = c2;    c2 = (a)[16];  (a)[16] = c1;   (a)[1] = c2; \
		c1 = (a)[6];   (a)[6] = (a)[3];   c2 = (a)[12];  (a)[12] = c1;  c1 = (a)[24]; \
		(a)[24] = c2;   c2 = (a)[17];  (a)[17] = c1;   (a)[3] = c2; \
		c1 = (a)[10];  (a)[10] = (a)[5];  c2 = (a)[20];  (a)[20] = c1;  c1 = (a)[9]; \
		(a)[9] = c2;    c2 = (a)[18];  (a)[18] = c1;   (a)[5] = c2; \
		c1 = (a)[14];  (a)[14] = (a)[7];  c2 = (a)[28];  (a)[28] = c1;  c1 = (a)[25]; \
		(a)[25] = c2;   c2 = (a)[19];  (a)[19] = c1;   (a)[7] = c2; \
		c1 = (a)[22];  (a)[22] = (a)[11]; c2 = (a)[13];  (a)[13] = c1;  c1 = (a)[26]; \
		(a)[26] = c2;   c2 = (a)[21];  (a)[21] = c1;   (a)[11] = c2; \
		c1 = (a)[30];  (a)[30] = (a)[15]; c2 = (a)[29];  (a)[29] = c1;  c1 = (a)[27]; \
		(a)[27] = c2;   c2 = (a)[23];  (a)[23] = c1;   (a)[15] = c2; \
	}

	#define fftKernel32(a, dir) \
	{ \
		fftKernel2S((a)[0],  (a)[16], dir); \
		fftKernel2S((a)[1],  (a)[17], dir); \
		fftKernel2S((a)[2],  (a)[18], dir); \
		fftKernel2S((a)[3],  (a)[19], dir); \
		fftKernel2S((a)[4],  (a)[20], dir); \
		fftKernel2S((a)[5],  (a)[21], dir); \
		fftKernel2S((a)[6],  (a)[22], dir); \
		fftKernel2S((a)[7],  (a)[23], dir); \
		fftKernel2S((a)[8],  (a)[24], dir); \
		fftKernel2S((a)[9],  (a)[25], dir); \
		fftKernel2S((a)[10], (a)[26], dir); \
		fftKernel2S((a)[11], (a)[27], dir); \
		fftKernel2S((a)[12], (a)[28], dir); \
		fftKernel2S((a)[13], (a)[29], dir); \
		fftKernel2S((a)[14], (a)[30], dir); \
		fftKernel2S((a)[15], (a)[31], dir); \
		(a)[17] = (a)[17] * complex_ctr(0x1.f6297cp-1f, dir*0x1.8f8b84p-3f); \
		(a)[18] = (a)[18] * complex_ctr(0x1.d906bcp-1f, dir*0x1.87de2ap-2f); \
		(a)[19] = (a)[19] * complex_ctr(0x1.a9b662p-1f, dir*0x1.1c73b4p-1f); \
		(a)[20] = (a)[20] * complex_ctr(0x1.6a09e6p-1f, dir*0x1.6a09e6p-1f); \
		(a)[21] = (a)[21] * complex_ctr(0x1.1c73b4p-1f, dir*0x1.a9b662p-1f); \
		(a)[22] = (a)[22] * complex_ctr(0x1.87de2ap-2f, dir*0x1.d906bcp-1f); \
		(a)[23] = (a)[23] * complex_ctr(0x1.8f8b84p-3f, dir*0x1.f6297cp-1f); \
		(a)[24] = (a)[24] * complex_ctr(0x0p+0f, dir*0x1p+0f); \
		(a)[25] = (a)[25] * complex_ctr(-0x1.8f8b84p-3f, dir*0x1.f6297cp-1f); \
		(a)[26] = (a)[26] * complex_ctr(-0x1.87de2ap-2f, dir*0x1.d906bcp-1f); \
		(a)[27] = (a)[27] * complex_ctr(-0x1.1c73b4p-1f, dir*0x1.a9b662p-1f); \
		(a)[28] = (a)[28] * complex_ctr(-0x1.6a09e6p-1f, dir*0x1.6a09e6p-1f); \
		(a)[29] = (a)[29] * complex_ctr(-0x1.a9b662p-1f, dir*0x1.1c73b4p-1f); \
		(a)[30] = (a)[30] * complex_ctr(-0x1.d906bcp-1f, dir*0x1.87de2ap-2f); \
		(a)[31] = (a)[31] * complex_ctr(-0x1.f6297cp-1f, dir*0x1.8f8b84p-3f); \
		fftKernel16((a), dir); \
		fftKernel16((a) + 16, dir); \
		bitreverse32((a)); \
	}
</%def>

<%def name="shiftGlobalBuffers(split)">
	%if split:
		in_real += offset;
		in_imag += offset;
		out_real += offset;
		out_imag += offset;
	%else:
		in += offset;
		out += offset;
	%endif
</%def>


<%def name="formattedLoad(a_index, g_index, split)">
	%if split:
		a[${a_index}].x = in_real[${g_index}];
		a[${a_index}].y = in_imag[${g_index}];
	%else:
		a[${a_index}] = in[${g_index}];
	%endif
</%def>

<%def name="formattedStore(a_index, g_index, split)">
	%if split:
		out_real[${g_index}] = a[${a_index}].x;
		out_imag[${g_index}] = a[${a_index}].y;
	%else:
		out[${g_index}] = a[${a_index}];
	%endif
</%def>

<%def name="insertGlobalLoadsAndTranspose(N, numWorkItemsPerXForm, numXFormsPerWG, R0, mem_coalesce_width, split)">

	<%
		log2NumWorkItemsPerXForm = log2(numWorkItemsPerXForm)
		groupSize = numWorkItemsPerXForm * numXFormsPerWG
	%>

	%if numXFormsPerWG > 1:
		s = S & ${numXFormsPerWG - 1};
	%endif

	%if numWorkItemsPerXForm >= mem_coalesce_width:
		%if numXFormsPerWG > 1:
			ii = lId & ${numWorkItemsPerXForm - 1};
			jj = lId >> ${log2NumWorkItemsPerXForm};

			if(!s || (groupId < gridDim.x - 1) || (jj < s))
			{
				offset = mad24(mad24(groupId, ${numXFormsPerWG}, jj), ${N}, ii);
				${shiftGlobalBuffers(split)}

			%for i in range(R0):
				${formattedLoad(i, i * numWorkItemsPerXForm, split)}
			%endfor
			}
		%else:
			ii = lId;
			jj = 0;
			offset = mad24(groupId, ${N}, ii);
			${shiftGlobalBuffers(split)}

			%for i in range(R0):
				${formattedLoad(i, i * numWorkItemsPerXForm, split)}
			%endfor
		%endif

	%elif N >= mem_coalesce_width:
		<%
			numInnerIter = N / mem_coalesce_width
			numOuterIter = numXFormsPerWG / (groupSize / mem_coalesce_width)
		%>

		ii = lId & ${mem_coalesce_width - 1};
		jj = lId >> ${log2(mem_coalesce_width)};
		lMemStore = mad24(jj, ${N + numWorkItemsPerXForm}, ii);
		offset = mad24(groupId, ${numXFormsPerWG}, jj);
		offset = mad24(offset, ${N}, ii);
		${shiftGlobalBuffers(split)}

		if((groupId == gridDim.x - 1) && s)
		{
		%for i in range(numOuterIter):
			if(jj < s)
			{
			%for j in range(numInnerIter):
				${formattedLoad(i * numInnerIter + j, \
					j * mem_coalesce_width + i * ( groupSize / mem_coalesce_width ) * N, split)}
			%endfor
			}
			%if i != numOuterIter - 1:
				jj += ${groupSize / mem_coalesce_width};
			%endif
		%endfor
		}
		else
		{
		%for i in range(numOuterIter):
			%for j in range(numInnerIter):
				${formattedLoad(i * numInnerIter + j, \
					j * mem_coalesce_width + i * ( groupSize / mem_coalesce_width ) * N, split)}
			%endfor
		%endfor
		}

		ii = lId & ${numWorkItemsPerXForm - 1};
		jj = lId >> ${log2NumWorkItemsPerXForm};
		lMemLoad = mad24(jj, ${N + numWorkItemsPerXForm}, ii);

		%for i in range(numOuterIter):
			%for j in range(numInnerIter):
				sMem[lMemStore + ${j * mem_coalesce_width + \
					i * (groupSize / mem_coalesce_width) * (N + numWorkItemsPerXForm)}] =
					a[${i * numInnerIter + j}].x;
			%endfor
		%endfor
		__syncthreads();

		%for i in range(R0):
			a[${i}].x = sMem[lMemLoad + ${i * numWorkItemsPerXForm}];
		%endfor
		__syncthreads();

		%for i in range(numOuterIter):
			%for j in range(numInnerIter):
				sMem[lMemStore + ${j * mem_coalesce_width + \
					i * (groupSize / mem_coalesce_width) * (N + numWorkItemsPerXForm )}] =
					a[${i * numInnerIter + j}].y;
			%endfor
		%endfor
		__syncthreads();

		%for i in range(R0):
			a[${i}].y = sMem[lMemLoad + ${i * numWorkItemsPerXForm}];
		%endfor
		__syncthreads();
	%else:
		offset = mad24(groupId, ${N * numXFormsPerWG}, lId);
		${shiftGlobalBuffers(split)}

		ii = lId & ${N - 1};
		jj = lId >> ${log2(N)};
		lMemStore = mad24(jj, ${N + numWorkItemsPerXForm}, ii);

		if((groupId == gridDim.x - 1) && s)
		{
		%for i in range(R0):
			if(jj < s)
				${formattedLoad(i, i * groupSize, split)}
			%if i != R0 - 1:
				jj += ${groupSize / N};
			%endif
		%endfor
		}
		else
		{
		%for i in range(R0):
			${formattedLoad(i, i*groupSize, split)}
		%endfor
		}

		%if numWorkItemsPerXForm > 1:
			ii = lId & ${numWorkItemsPerXForm - 1};
			jj = lId >> ${log2NumWorkItemsPerXForm};
			lMemLoad = mad24(jj, ${N + numWorkItemsPerXForm}, ii);
		%else:
			ii = 0;
			jj = lId;
			lMemLoad = mul24(jj, ${N + numWorkItemsPerXForm});
		%endif

		%for i in range(R0):
			sMem[lMemStore + ${i * ( groupSize / N ) * ( N + numWorkItemsPerXForm )}] = a[${i}].x;
		%endfor
		__syncthreads();

		%for i in range(R0):
			a[${i}].x = sMem[lMemLoad + ${i * numWorkItemsPerXForm}];
		%endfor
		__syncthreads();

		%for i in range(R0):
			sMem[lMemStore + ${i * (groupSize / N) * (N + numWorkItemsPerXForm)}] = a[${i}].y;
		%endfor
		__syncthreads();

		%for i in range(R0):
			a[${i}].y = sMem[lMemLoad + ${i * numWorkItemsPerXForm}];
		%endfor
		__syncthreads();
	%endif
</%def>

<%def name="insertGlobalStoresAndTranspose(N, maxRadix, Nr, numWorkItemsPerXForm, numXFormsPerWG, mem_coalesce_width, split)">

	<%
		groupSize = numWorkItemsPerXForm * numXFormsPerWG
		numIter = maxRadix / Nr
	%>

	%if numWorkItemsPerXForm >= mem_coalesce_width:
		%if numXFormsPerWG > 1:
			if(!s || (groupId < gridDim.x - 1) || (jj < s))
			{
		%endif

		%for i in range(maxRadix):
			<%
				j = i % numIter
				k = i / numIter
				ind = j * Nr + k
			%>
			${formattedStore(ind, i * numWorkItemsPerXForm, split)}
		%endfor

		%if numXFormsPerWG > 1:
			}
		%endif

	%elif N >= mem_coalesce_width:
		<%
			numInnerIter = N / mem_coalesce_width
			numOuterIter = numXFormsPerWG / (groupSize / mem_coalesce_width)
		%>
		lMemLoad  = mad24(jj, ${N + numWorkItemsPerXForm}, ii);
		ii = lId & ${mem_coalesce_width - 1};
		jj = lId >> ${log2(mem_coalesce_width)};
		lMemStore = mad24(jj, ${N + numWorkItemsPerXForm}, ii);

		%for i in range(maxRadix):
			<%
				j = i % numIter
				k = i / numIter
				ind = j * Nr + k
			%>
			sMem[lMemLoad + ${i * numWorkItemsPerXForm}] = a[${ind}].x;
		%endfor
		__syncthreads();

		%for i in range(numOuterIter):
			%for j in range(numInnerIter):
				a[${i*numInnerIter + j}].x = sMem[lMemStore + ${j * mem_coalesce_width + \
					i * (groupSize / mem_coalesce_width) * (N + numWorkItemsPerXForm)}];
			%endfor
		%endfor
		__syncthreads();

		%for i in range(maxRadix):
			<%
				j = i % numIter
				k = i / numIter
				ind = j * Nr + k
			%>
			sMem[lMemLoad + ${i * numWorkItemsPerXForm}] = a[${ind}].y;
		%endfor
		__syncthreads();

		%for i in range(numOuterIter):
			%for j in range(numInnerIter):
				a[${i * numInnerIter + j}].y = sMem[lMemStore + ${j * mem_coalesce_width + \
					i * (groupSize / mem_coalesce_width) * (N + numWorkItemsPerXForm)}];
			%endfor
		%endfor
		__syncthreads();

		if((groupId == gridDim.x - 1) && s)
		{
		%for i in range(numOuterIter):
			if(jj < s)
			{
			%for j in range(numInnerIter):
				${formattedStore(i * numInnerIter + j, \
					j * mem_coalesce_width + i * (groupSize / mem_coalesce_width) * N, split)}
			%endfor
			}
			%if i != numOuterIter - 1:
				jj += ${groupSize / mem_coalesce_width};
			%endif
		%endfor
		}
		else
		{
		%for i in range(numOuterIter):
			%for j in range(numInnerIter):
				${formattedStore(i * numInnerIter + j, \
					j * mem_coalesce_width + i * (groupSize / mem_coalesce_width) * N, split)}
			%endfor
		%endfor
		}
	%else:
		lMemLoad = mad24(jj, ${N + numWorkItemsPerXForm}, ii);
		ii = lId & ${N - 1};
		jj = lId >> ${log2(N)};
		lMemStore = mad24(jj, ${N + numWorkItemsPerXForm}, ii);

		%for i in range(maxRadix):
			<%
				j = i % numIter
				k = i / numIter
				ind = j * Nr + k
			%>
			sMem[lMemLoad + ${i * numWorkItemsPerXForm}] = a[${ind}].x;
		%endfor
		__syncthreads();

		%for i in range(maxRadix):
			a[${i}].x = sMem[lMemStore + ${i * (groupSize / N) * (N + numWorkItemsPerXForm)}];
		%endfor
		__syncthreads();

		%for i in range(maxRadix):
			<%
				j = i % numIter
				k = i / numIter
				ind = j * Nr + k
			%>
			sMem[lMemLoad + ${i * numWorkItemsPerXForm}] = a[${ind}].y;
		%endfor
		__syncthreads();

		%for i in range(maxRadix):
			a[${i}].y = sMem[lMemStore + ${i * (groupSize / N) * (N + numWorkItemsPerXForm)}];
		%endfor
		__syncthreads();

		if((groupId == gridDim.x - 1) && s)
		{
		%for i in range(maxRadix):
			if(jj < s )
			{
				${formattedStore(i, i * groupSize, split)}
			}
			%if i != maxRadix - 1:
				jj += ${groupSize / N};
			%endif
		%endfor
		}
		else
		{
			%for i in range(maxRadix):
				${formattedStore(i, i * groupSize, split)}
			%endfor
		}
	%endif
</%def>

<%def name="insertfftKernel(Nr, numIter)">
	%for i in range(numIter):
		fftKernel${Nr}(a + ${i * Nr}, dir);
	%endfor
</%def>

<%def name="insertTwiddleKernel(Nr, numIter, Nprev, len, numWorkItemsPerXForm, scalar)">

	<% logNPrev = log2(Nprev) %>

	%for z in range(numIter):
		%if z == 0:
			%if Nprev > 1:
				angf = (${scalar})(ii >> ${logNPrev});
			%else:
				angf = (${scalar})ii;
			%endif
		%else:
			%if Nprev > 1:
				angf = (${scalar})((${z * numWorkItemsPerXForm} + ii) >> ${logNPrev});
			%else:
				## TODO: find out which conditions are necessary to execute this code
				angf = (${scalar})(${z * numWorkItemsPerXForm} + ii);
			%endif
		%endif

		%for k in range(1, Nr):
			<% ind = z * Nr + k %>
			ang = dir * ((${scalar})2 * M_PI * (${scalar})${k} / (${scalar})${len}) * angf;
			## TODO: use native_cos and sin (as OpenCL implementation did)
			w = make_float2(cos(ang), sin(ang));
			a[${ind}] = a[${ind}] * w;
		%endfor
	%endfor
</%def>

<%def name="insertLocalStores(numIter, Nr, numWorkItemsPerXForm, numWorkItemsReq, offset, comp)">
	%for z in range(numIter):
		%for k in range(Nr):
			<% index = k * (numWorkItemsReq + offset) + z * numWorkItemsPerXForm %>
			sMem[lMemStore + ${index}] = a[${z * Nr + k}].${comp};
		%endfor
	%endfor
	__syncthreads();
</%def>

<%def name="insertLocalLoads(n, Nr, Nrn, Nprev, Ncurr, numWorkItemsPerXForm, numWorkItemsReq, offset, comp)">
	<%
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
	%>

	%for i in range(iter):
		<%
			ii = i / (interBlockHNum * vertNum)
			zz = i % (interBlockHNum * vertNum)
			jj = zz % interBlockHNum
			kk = zz / interBlockHNum
		%>

		%for z in range(Nrn):
			<% st = kk * vertStride + jj * interBlockHStride + ii * intraBlockHStride + z * stride %>
			a[${i * Nrn + z}].${comp} = sMem[lMemLoad + ${st}];
		%endfor
	%endfor
	__syncthreads();
</%def>

<%def name="insertLocalLoadIndexArithmatic(Nprev, Nr, numWorkItemsReq, numWorkItemsPerXForm, numXFormsPerWG, offset, midPad)">
	<%
		Ncurr = Nprev * Nr
		logNcurr = log2(Ncurr)
		logNprev = log2(Nprev)
		incr = (numWorkItemsReq + offset) * Nr + midPad
	%>

	%if Ncurr < numWorkItemsPerXForm:
		%if Nprev == 1:
			j = ii & ${Ncurr - 1};
		%else:
			j = (ii & ${Ncurr - 1}) >> ${logNprev};
		%endif

		%if Nprev == 1:
			i = ii >> ${logNcurr};
		%else:
			i = mad24(ii >> ${logNcurr}, ${Nprev}, ii & ${Nprev - 1});
		%endif
	%else:
		%if Nprev == 1:
			j = ii;
		%else:
			j = ii >> ${logNprev};
		%endif

		%if Nprev == 1:
			i = 0;
		%else:
			i = ii & ${Nprev - 1};
		%endif
	%endif

	%if numXFormsPerWG > 1:
		i = mad24(jj, ${incr}, i);
	%endif

	lMemLoad = mad24(j, ${numWorkItemsReq + offset}, i);
</%def>

<%def name="insertLocalStoreIndexArithmatic(numWorkItemsReq, numXFormsPerWG, Nr, offset, midPad)">
	%if numXFormsPerWG == 1:
		lMemStore = ii;
	%else:
		lMemStore = mad24(jj, ${(numWorkItemsReq + offset) * Nr + midPad}, ii);
	%endif
</%def>

<%def name="localKernel(scalar, complex, split, kernel_name, shared_mem, numWorkItemsPerXForm, numXFormsPerWG, \
	min_mem_coalesce_width, N, n, num_local_mem_banks, log2, getPadding)">

	<% max_radix = N[0] %>

${baseKernels(scalar, complex)}

extern "C" {

%if split:
	__global__ void ${kernel_name}(${scalar} *in_real, ${scalar} *in_imag, ${scalar} *out_real, ${scalar} *out_imag, int dir, int S)
%else:
	__global__ void ${kernel_name}(${complex} *in, ${complex} *out, int dir, int S)
%endif
	{
		%if shared_mem > 0:
			__shared__ float sMem[${shared_mem}];
		%endif

		int i, j, r, indexIn, indexOut, index, tid, bNum, xNum, k, l;
		int s, ii, jj, offset;
		${complex} w;
		${scalar} ang, angf, ang1;
		size_t lMemStore, lMemLoad;

		// need to fill a[] with zeros, because otherwise nvcc crashes
		// (it considers a[] not initialized)
		${complex} a[${max_radix}] = {${', '.join(['0'] * max_radix * 2)}};

		int lId = threadIdx.x;
		int groupId = blockIdx.x;

		${insertGlobalLoadsAndTranspose(n, numWorkItemsPerXForm, numXFormsPerWG, max_radix,
			min_mem_coalesce_width, split)}

		<%
			Nprev = 1
			len_ = n
			numRadix = len(N)
		%>

	%for r in range(numRadix):
		<%
			numIter = N[0] / N[r]
			numWorkItemsReq = n / N[r]
			Ncurr = Nprev * N[r]
		%>

		${insertfftKernel(N[r], numIter)}

		%if r < numRadix - 1:
			${insertTwiddleKernel(N[r], numIter, Nprev, len_, numWorkItemsPerXForm, scalar)}
			<%
				lMemSize, offset, midPad = getPadding(numWorkItemsPerXForm, Nprev, numWorkItemsReq,
					numXFormsPerWG, N[r], num_local_mem_banks)
			%>
			${insertLocalStoreIndexArithmatic(numWorkItemsReq, numXFormsPerWG, N[r], offset, midPad)}
			${insertLocalLoadIndexArithmatic(Nprev, N[r], numWorkItemsReq, numWorkItemsPerXForm, numXFormsPerWG, offset, midPad)}
			${insertLocalStores(numIter, N[r], numWorkItemsPerXForm, numWorkItemsReq, offset, "x")}
			${insertLocalLoads(n, N[r], N[r+1], Nprev, Ncurr, numWorkItemsPerXForm, numWorkItemsReq, offset, "x")}
			${insertLocalStores(numIter, N[r], numWorkItemsPerXForm, numWorkItemsReq, offset, "y")}
			${insertLocalLoads(n, N[r], N[r+1], Nprev, Ncurr, numWorkItemsPerXForm, numWorkItemsReq, offset, "y")}
			<%
				Nprev = Ncurr
				len_ = len_ / N[r]
			%>
		%endif
	%endfor

	${insertGlobalStoresAndTranspose(n, max_radix, N[numRadix - 1], numWorkItemsPerXForm,
		numXFormsPerWG, min_mem_coalesce_width, split)}

	}
}

</%def>

<%def name="globalKernel(scalar, complex, split, passNum, kernel_name, radixArr, numPasses, shared_mem, R1Arr, \
	R2Arr, Rinit, batchSize, BS, vertBS, vertical, maxThreadsPerBlock, max_work_item_per_workgroup, n, N, log2, getPadding)">

${baseKernels(scalar, complex)}

extern "C" {
	<%
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
		threadsPerBlock = batchSize * threadsPerXForm
		threadsPerBlock = min(threadsPerBlock, maxThreadsPerBlock)

		numIter = R1 / R2
		gInInc = threadsPerBlock / batchSize
		lgStrideO = log2(strideO)
		numBlocksPerXForm = strideI / batchSize
		numBlocks = numBlocksPerXForm
		if not vertical:
			numBlocks *= BS
		else:
			numBlocks *= vertBS

		m = log2(n)
	%>
	// ${strideI} ${batchSize} ${passNum} ${str(radixArr)}
%if split:
	__global__ void ${kernel_name}(${scalar} *in_real, ${scalar} *in_imag, ${scalar} *out_real, ${scalar} *out_imag, int dir, int S)
%else:
	__global__ void ${kernel_name}(${complex} *in, ${complex} *out, int dir, int S)
%endif
	{
		%if shared_mem > 0:
			__shared__ float sMem[${shared_mem}];
		%endif

		int i, j, r, indexIn, indexOut, index, tid, bNum, xNum, k, l;
		int s, ii, jj, offset;
		${complex} w;
		${scalar} ang, angf, ang1;
		size_t lMemStore, lMemLoad;

		// need to fill a[] with zeros, because otherwise nvcc crashes
		// (it considers a[] not initialized)
		${complex} a[${R1}] = {${', '.join(['0'] * R1 * 2)}};

		int lId = threadIdx.x;
		int groupId = blockIdx.x;

		%if vertical:
			xNum = groupId >> ${log2(numBlocksPerXForm)};
			groupId = groupId & ${numBlocksPerXForm - 1};
			indexIn = mad24(groupId, ${batchSize}, xNum << ${log2(n * BS)});
			tid = mul24(groupId, ${batchSize});
			i = tid >> ${lgStrideO};
			j = tid & ${strideO - 1};
			<%
				stride = radix * Rinit
				for i in range(passNum):
					stride *= radixArr[i]
			%>
			indexOut = mad24(i, ${stride}, j + (xNum << ${log2(n*BS)}));
			bNum = groupId;
		%else:
			<% lgNumBlocksPerXForm = log2(numBlocksPerXForm) %>
			bNum = groupId & ${numBlocksPerXForm - 1};
			xNum = groupId >> ${lgNumBlocksPerXForm};
			indexIn = mul24(bNum, ${batchSize});
			tid = indexIn;
			i = tid >> ${lgStrideO};
			j = tid & ${strideO - 1};
			<%
				stride = radix*Rinit
				for i in range(passNum):
					stride *= radixArr[i]
			%>
			indexOut = mad24(i, ${stride}, j);
			indexIn += (xNum << ${m});
			indexOut += (xNum << ${m});
		%endif

		## Load Data
		<% lgBatchSize = log2(batchSize) %>
		tid = lId;
		i = tid & ${batchSize - 1};
		j = tid >> ${lgBatchSize};
		indexIn += mad24(j, ${strideI}, i);

		%if split:
			in_real += indexIn;
			in_imag += indexIn;
			%for j in range(R1):
				a[${j}].x = in_real[${j * gInInc * strideI}];
			%endfor
			%for j in range(R1):
				a[${j}].y = in_imag[${j * gInInc * strideI}];
			%endfor
		%else:
			in += indexIn;
			%for j in range(R1):
				a[${j}] = in[${j * gInInc * strideI}];
			%endfor
		%endif

		fftKernel${R1}(a, dir);

		%if R2 > 1:
			## twiddle
			%for k in range(1, R1):
				ang = dir * ((${scalar})2 * M_PI * ${k} / ${radix}) * j;
				## TODO: use native cos and sin (as OpenCL implementation did)
				w = complex_ctr(cos(ang), sin(ang));
				a[${k}] = a[${k}] * w;
			%endfor

			## shuffle
			<% numIter = R1 / R2 %>
			indexIn = mad24(j, ${threadsPerBlock * numIter}, i);
			lMemStore = tid;
			lMemLoad = indexIn;

			%for k in range(R1):
				sMem[lMemStore + ${k * threadsPerBlock}] = a[${k}].x;
			%endfor
			__syncthreads();

			%for k in range(numIter):
				%for t in range(R2):
					a[${k * R2 + t}].x = sMem[lMemLoad + ${t * batchSize + k * threadsPerBlock}];
				%endfor
			%endfor
			__syncthreads();

			%for k in range(R1):
				sMem[lMemStore + ${k * threadsPerBlock}] = a[${k}].y;
			%endfor
			__syncthreads();

			%for k in range(numIter):
				%for t in range(R2):
					a[${k * R2 + t}].y = sMem[lMemLoad + ${t * batchSize + k * threadsPerBlock}];
				%endfor
			%endfor
			__syncthreads();

			%for j in range(numIter):
				fftKernel${R2}(a + ${j * R2}, dir);
			%endfor
		%endif

		## twiddle
		%if passNum < numPasses - 1:
			l = ((bNum << ${lgBatchSize}) + i) >> ${lgStrideO};
			k = j << ${log2(R1 / R2)};
			ang1 = dir * ((${scalar})2 * M_PI / ${N}) * l;
			%for t in range(R1):
				ang = ang1 * (k + ${(t % R2) * R1 + (t / R2)});
				## TODO: use native cos and sin (as OpenCL implementation did)
				w = complex_ctr(cos(ang), sin(ang));
				a[${t}] = a[${t}] * w;
			%endfor
		%endif

		## Store Data
		%if strideO == 1:
			lMemStore = mad24(i, ${radix + 1}, j << ${log2(R1 / R2)});
			lMemLoad = mad24(tid >> ${log2(radix)}, ${radix + 1}, tid & ${radix - 1});

			%for i in range(R1 / R2):
				%for j in range(R2):
					sMem[lMemStore + ${i + j*R1}] = a[${i * R2 + j}].x;
				%endfor
			%endfor
			__syncthreads();

			%for i in range(R1):
				a[${i}].x = sMem[lMemLoad + ${i * (radix + 1) * (threadsPerBlock / radix)}];
			%endfor
			__syncthreads();

			%for i in range(R1/R2):
				%for j in range(R2):
					sMem[lMemStore + ${i + j*R1}] = a[${i * R2 + j}].y;
				%endfor
			%endfor
			__syncthreads();

			%for i in range(R1):
				a[${i}].y = sMem[lMemLoad + ${i * (radix + 1) * (threadsPerBlock / radix)}];
			%endfor
			__syncthreads();

			indexOut += tid;

			%if split:
				out_real += indexOut;
				out_imag += indexOut;
				%for k in range(R1):
					out_real[${k * threadsPerBlock}] = a[${k}].x;
				%endfor
				%for k in range(R1):
					out_imag[${k * threadsPerBlock}] = a[${k}].y;
				%endfor
			%else:
				out += indexOut;
				%for k in range(R1):
					out[${k * threadsPerBlock}] = a[${k}];
				%endfor
			%endif
		%else:
			indexOut += mad24(j, ${numIter * strideO}, i);
			%if split:
				out_real += indexOut;
				out_imag += indexOut;
				%for k in range(R1):
					out_real[${((k % R2) * R1 + (k / R2)) * strideO}] = a[${k}].x;
				%endfor
				%for k in range(R1):
					out_imag[${((k % R2) * R1 + (k / R2)) * strideO}] = a[${k}].y;
				%endfor
			%else:
				out += indexOut;
				%for k in range(R1):
					out[${((k % R2) * R1 + (k / R2)) * strideO}] = a[${k}];
				%endfor
			%endif
		%endif
	}

}

</%def>
