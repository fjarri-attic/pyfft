<%!
	import math
%>

<%def name="insertBaseKernels(scalar, complex)">

	#define complex_ctr(x, y) make_${complex}(x, y)

	## TODO: replace by intrinsincs if necessary

	## multiplication + addition
	#define mad24(x, y, z) ((x) * (y) + (z))
	#define mad(x, y, z) ((x) * (y) + (z))

	## integer multiplication
	#define mul24(x, y) __mul24(x, y)

	## TODO: add double-precision support
	#define complex_exp(res, ang) __sincosf(ang, &(res.y), &(res.x))

	inline ${complex} operator+(${complex} a, ${complex} b) { return complex_ctr(a.x + b.x, a.y + b.y); }
	inline ${complex} operator-(${complex} a, ${complex} b) { return complex_ctr(a.x - b.x, a.y - b.y); }
	inline ${complex} operator*(${complex} a, ${scalar}  b) { return complex_ctr(b * a.x, b * a.y); }
	inline ${complex} operator/(${complex} a, int b) { return complex_ctr(a.x / b, a.y / b); }
	inline ${complex} operator*(${complex} a, ${complex} b)
	{
		return complex_ctr(mad(-a.y, b.y, a.x * b.x), mad(a.y, b.x, a.x * b.y));
	}

	#define conj(a) complex_ctr((a).x, -(a).y)
	#define conjTransp(a) complex_ctr(-(a).y, (a).x)

	template<class T>
	__device__ void swap(T &a, T &b)
	{
		T c = a;
		a = b;
		b = c;
	}

	// shifts the sequence (a1, a2, a3, a4, a5) transforming it to
	// (a5, a1, a2, a3, a4)
	template<class T>
	__device__ void shift(T &a1, T &a2, T &a3, T &a4, T &a5)
	{
		T c1, c2;
		c1 = a2;
		a2 = a1;
		c2 = a3;
		a3 = c1;
		c1 = a4;
		a4 = c2;
		c2 = a5;
		a5 = c1;
		a1 = c2;
	}

	template<int dir>
	__device__ void fftKernel2(${complex} *a)
	{
		${complex} c = a[0];
		a[0] = c + a[1];
		a[1] = c - a[1];
	}

	template<int dir>
	__device__ void fftKernel2S(${complex} &d1, ${complex} &d2)
	{
		${complex} c = d1;
		d1 = c + d2;
		d2 = c - d2;
	}

	template<int dir>
	__device__ void fftKernel4(${complex} *a)
	{
		fftKernel2S<dir>(a[0], a[2]);
		fftKernel2S<dir>(a[1], a[3]);
		fftKernel2S<dir>(a[0], a[1]);
		a[3] = conjTransp(a[3]) * dir;
		fftKernel2S<dir>(a[2], a[3]);
		swap(a[1], a[2]);
	}

	template<int dir>
	__device__ void fftKernel4s(${complex} &a0, ${complex} &a1, ${complex} &a2, ${complex} &a3)
	{
		fftKernel2S<dir>(a0, a2);
		fftKernel2S<dir>(a1, a3);
		fftKernel2S<dir>(a0, a1);
		(a3) = conjTransp(a3) * dir;
		fftKernel2S<dir>(a2, a3);
		swap(a1, a2);
	}

	__device__ void bitreverse8(${complex} *a)
	{
		swap(a[1], a[4]);
		swap(a[3], a[6]);
	}

	template<int dir>
	__device__ void fftKernel8(${complex} *a)
	{
		const ${complex} w1  = complex_ctr((${scalar})${math.sin(math.pi / 4)}, (${scalar})${math.sin(math.pi / 4)} * dir);
		const ${complex} w3  = complex_ctr((${scalar})-${math.sin(math.pi / 4)}, (${scalar})${math.sin(math.pi / 4)} * dir);
		fftKernel2S<dir>(a[0], a[4]);
		fftKernel2S<dir>(a[1], a[5]);
		fftKernel2S<dir>(a[2], a[6]);
		fftKernel2S<dir>(a[3], a[7]);
		a[5] = w1 * a[5];
		a[6] = conjTransp(a[6]) * dir;
		a[7] = w3 * a[7];
		fftKernel2S<dir>(a[0], a[2]);
		fftKernel2S<dir>(a[1], a[3]);
		fftKernel2S<dir>(a[4], a[6]);
		fftKernel2S<dir>(a[5], a[7]);
		a[3] = conjTransp(a[3]) * dir;
		a[7] = conjTransp(a[7]) * dir;
		fftKernel2S<dir>(a[0], a[1]);
		fftKernel2S<dir>(a[2], a[3]);
		fftKernel2S<dir>(a[4], a[5]);
		fftKernel2S<dir>(a[6], a[7]);
		bitreverse8(a);
	}

	__device__ void bitreverse4x4(${complex} *a)
	{
		swap(a[1], a[4]);
		swap(a[2], a[8]);
		swap(a[3], a[12]);
		swap(a[6], a[9]);
		swap(a[7], a[13]);
		swap(a[11], a[14]);
	}

	template<int dir>
	__device__ void fftKernel16(${complex} *a)
	{
		const ${scalar} w0 = (${scalar})${math.cos(math.pi / 8)};
		const ${scalar} w1 = (${scalar})${math.sin(math.pi / 8)};
		const ${scalar} w2 = (${scalar})${math.sin(math.pi / 4)};
		fftKernel4s<dir>(a[0], a[4], a[8],  a[12]);
		fftKernel4s<dir>(a[1], a[5], a[9],  a[13]);
		fftKernel4s<dir>(a[2], a[6], a[10], a[14]);
		fftKernel4s<dir>(a[3], a[7], a[11], a[15]);
		a[5]  = a[5] * complex_ctr(w0, dir * w1);
		a[6]  = a[6] * complex_ctr(w2, dir * w2);
		a[7]  = a[7] * complex_ctr(w1, dir * w0);
		a[9]  = a[9] * complex_ctr(w2, dir * w2);
		a[10] = complex_ctr(dir, 0) * (conjTransp(a[10]));
		a[11] = a[11] * complex_ctr(-w2, dir * w2);
		a[13] = a[13] * complex_ctr(w1, dir * w0);
		a[14] = a[14] * complex_ctr(-w2, dir * w2);
		a[15] = a[15] * complex_ctr(-w0, -dir * w1);
		fftKernel4<dir>(a);
		fftKernel4<dir>(a + 4);
		fftKernel4<dir>(a + 8);
		fftKernel4<dir>(a + 12);
		bitreverse4x4(a);
	}

	__device__ void bitreverse32(${complex} *a)
	{
		shift(a[1], a[2], a[4], a[8], a[16]);
		shift(a[3], a[6], a[12], a[24], a[17]);
		shift(a[5], a[10], a[20], a[9], a[18]);
		shift(a[7], a[14], a[28], a[25], a[19]);
		shift(a[11], a[22], a[13], a[26], a[21]);
		shift(a[15], a[30], a[29], a[27], a[23]);
	}

	template<int dir>
	__device__ void fftKernel32(${complex} *a)
	{
		%for i in range(16):
			fftKernel2S<dir>(a[${i}], a[${i + 16}]);
		%endfor

		%for i in range(1, 16):
			a[${i + 16}] = a[${i + 16}] * complex_ctr(
				(${scalar})${math.cos(i * math.pi / 16)},
				(${scalar})${math.sin(i * math.pi / 16)}
			);
		%endfor

		fftKernel16<dir>(a);
		fftKernel16<dir>(a + 16);
		bitreverse32(a);
	}
</%def>

<%def name="insertGlobalBuffersShift(split)">
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


<%def name="insertGlobalLoad(a_index, g_index, split)">
	%if split:
		a[${a_index}].x = in_real[${g_index}];
		a[${a_index}].y = in_imag[${g_index}];
	%else:
		a[${a_index}] = in[${g_index}];
	%endif
</%def>

<%def name="insertGlobalStore(a_index, g_index, split, normalization_coeff)">
	%if split:
		out_real[${g_index}] = a[${a_index}].x / (dir == 1 ? ${normalization_coeff} : 1);
		out_imag[${g_index}] = a[${a_index}].y / (dir == 1 ? ${normalization_coeff} : 1);
	%else:
		out[${g_index}] = a[${a_index}] / (dir == 1 ? ${normalization_coeff} : 1);
	%endif
</%def>

<%def name="insertGlobalLoadsAndTranspose(n, threads_per_xform, xforms_per_block, radix, mem_coalesce_width, split)">

	<%
		log2_threads_per_xform = log2(threads_per_xform)
		block_size = threads_per_xform * xforms_per_block
	%>

	%if xforms_per_block > 1:
		s = S & ${xforms_per_block - 1};
	%endif

	%if threads_per_xform >= mem_coalesce_width:
		%if xforms_per_block > 1:
			ii = thread_id & ${threads_per_xform - 1};
			jj = thread_id >> ${log2_threads_per_xform};

			if(!s || (block_id < gridDim.x - 1) || (jj < s))
			{
				{
					int offset = mad24(mad24(block_id, ${xforms_per_block}, jj), ${n}, ii);
					${insertGlobalBuffersShift(split)}
				}

			%for i in range(radix):
				${insertGlobalLoad(i, i * threads_per_xform, split)}
			%endfor
			}
		%else:
			ii = thread_id;

			{
				int offset = mad24(block_id, ${n}, ii);
				${insertGlobalBuffersShift(split)}
			}

			%for i in range(radix):
				${insertGlobalLoad(i, i * threads_per_xform, split)}
			%endfor
		%endif

	%elif n >= mem_coalesce_width:
		<%
			num_inner_iter = n / mem_coalesce_width
			num_outer_iter = xforms_per_block / (block_size / mem_coalesce_width)
		%>

		ii = thread_id & ${mem_coalesce_width - 1};
		jj = thread_id >> ${log2(mem_coalesce_width)};
		smem_store_index = mad24(jj, ${n + threads_per_xform}, ii);

		{
			int offset = mad24(block_id, ${xforms_per_block}, jj);
			offset = mad24(offset, ${n}, ii);
			${insertGlobalBuffersShift(split)}
		}

		if((block_id == gridDim.x - 1) && s)
		{
		%for i in range(num_outer_iter):
			if(jj < s)
			{
			%for j in range(num_inner_iter):
				${insertGlobalLoad(i * num_inner_iter + j, \
					j * mem_coalesce_width + i * (block_size / mem_coalesce_width) * n, split)}
			%endfor
			}
			%if i != num_outer_iter - 1:
				jj += ${block_size / mem_coalesce_width};
			%endif
		%endfor
		}
		else
		{
		%for i in range(num_outer_iter):
			%for j in range(num_inner_iter):
				${insertGlobalLoad(i * num_inner_iter + j, \
					j * mem_coalesce_width + i * (block_size / mem_coalesce_width) * n, split)}
			%endfor
		%endfor
		}

		ii = thread_id & ${threads_per_xform - 1};
		jj = thread_id >> ${log2_threads_per_xform};
		smem_load_index = mad24(jj, ${n + threads_per_xform}, ii);

		%for comp in ('x', 'y'):
			%for i in range(num_outer_iter):
				%for j in range(num_inner_iter):
					smem[smem_store_index + ${j * mem_coalesce_width + \
						i * (block_size / mem_coalesce_width) * (n + threads_per_xform)}] =
						a[${i * num_inner_iter + j}].${comp};
				%endfor
			%endfor
			__syncthreads();

			%for i in range(radix):
				a[${i}].${comp} = smem[smem_load_index + ${i * threads_per_xform}];
			%endfor
			__syncthreads();
		%endfor
	%else:
		{
			int offset = mad24(block_id, ${n * xforms_per_block}, thread_id);
			${insertGlobalBuffersShift(split)}
		}

		ii = thread_id & ${n - 1};
		jj = thread_id >> ${log2(n)};
		smem_store_index = mad24(jj, ${n + threads_per_xform}, ii);

		if((block_id == gridDim.x - 1) && s)
		{
		%for i in range(radix):
			if(jj < s)
				${insertGlobalLoad(i, i * block_size, split)}
			%if i != radix - 1:
				jj += ${block_size / n};
			%endif
		%endfor
		}
		else
		{
		%for i in range(radix):
			${insertGlobalLoad(i, i*block_size, split)}
		%endfor
		}

		%if threads_per_xform > 1:
			ii = thread_id & ${threads_per_xform - 1};
			jj = thread_id >> ${log2_threads_per_xform};
			smem_load_index = mad24(jj, ${n + threads_per_xform}, ii);
		%else:
			ii = 0;
			jj = thread_id;
			smem_load_index = mul24(jj, ${n + threads_per_xform});
		%endif

		%for comp in ('x', 'y'):
			%for i in range(radix):
				smem[smem_store_index + ${i * (block_size / n) * (n + threads_per_xform)}] = a[${i}].${comp};
			%endfor
			__syncthreads();

			%for i in range(radix):
				a[${i}].${comp} = smem[smem_load_index + ${i * threads_per_xform}];
			%endfor
			__syncthreads();
		%endfor
	%endif
</%def>

<%def name="insertGlobalStoresAndTranspose(n, max_radix, radix, threads_per_xform, xforms_per_block, \
	   mem_coalesce_width, split, normalization_coeff)">

	<%
		block_size = threads_per_xform * xforms_per_block
		num_iter = max_radix / radix
	%>

	%if threads_per_xform >= mem_coalesce_width:
		%if xforms_per_block > 1:
			if(!s || (block_id < gridDim.x - 1) || (jj < s))
			{
		%endif

		%for i in range(max_radix):
			<%
				j = i % num_iter
				k = i / num_iter
				ind = j * radix + k
			%>
			${insertGlobalStore(ind, i * threads_per_xform, split, normalization_coeff)}
		%endfor

		%if xforms_per_block > 1:
			}
		%endif

	%elif n >= mem_coalesce_width:
		<%
			num_inner_iter = n / mem_coalesce_width
			num_outer_iter = xforms_per_block / (block_size / mem_coalesce_width)
		%>
		smem_load_index  = mad24(jj, ${n + threads_per_xform}, ii);
		ii = thread_id & ${mem_coalesce_width - 1};
		jj = thread_id >> ${log2(mem_coalesce_width)};
		smem_store_index = mad24(jj, ${n + threads_per_xform}, ii);

		%for comp in ('x', 'y'):
			%for i in range(max_radix):
				<%
					j = i % num_iter
					k = i / num_iter
					ind = j * radix + k
				%>
				smem[smem_load_index + ${i * threads_per_xform}] = a[${ind}].${comp};
			%endfor
			__syncthreads();

			%for i in range(num_outer_iter):
				%for j in range(num_inner_iter):
					a[${i*num_inner_iter + j}].${comp} = smem[smem_store_index + ${j * mem_coalesce_width + \
						i * (block_size / mem_coalesce_width) * (n + threads_per_xform)}];
				%endfor
			%endfor
			__syncthreads();
		%endfor

		if((block_id == gridDim.x - 1) && s)
		{
		%for i in range(num_outer_iter):
			if(jj < s)
			{
			%for j in range(num_inner_iter):
				${insertGlobalStore(i * num_inner_iter + j, \
					j * mem_coalesce_width + i * (block_size / mem_coalesce_width) * n, \
					split, normalization_coeff)}
			%endfor
			}
			%if i != num_outer_iter - 1:
				jj += ${block_size / mem_coalesce_width};
			%endif
		%endfor
		}
		else
		{
		%for i in range(num_outer_iter):
			%for j in range(num_inner_iter):
				${insertGlobalStore(i * num_inner_iter + j, \
					j * mem_coalesce_width + i * (block_size / mem_coalesce_width) * n, \
					split, normalization_coeff)}
			%endfor
		%endfor
		}
	%else:
		smem_load_index = mad24(jj, ${n + threads_per_xform}, ii);
		ii = thread_id & ${n - 1};
		jj = thread_id >> ${log2(n)};
		smem_store_index = mad24(jj, ${n + threads_per_xform}, ii);

		%for comp in ('x', 'y'):
			%for i in range(max_radix):
				<%
					j = i % num_iter
					k = i / num_iter
					ind = j * radix + k
				%>
				smem[smem_load_index + ${i * threads_per_xform}] = a[${ind}].${comp};
			%endfor
			__syncthreads();

			%for i in range(max_radix):
				a[${i}].${comp} = smem[smem_store_index + ${i * (block_size / n) * (n + threads_per_xform)}];
			%endfor
			__syncthreads();
		%endfor

		if((block_id == gridDim.x - 1) && s)
		{
		%for i in range(max_radix):
			if(jj < s)
			{
				${insertGlobalStore(i, i * block_size, split, normalization_coeff)}
			}
			%if i != max_radix - 1:
				jj += ${block_size / n};
			%endif
		%endfor
		}
		else
		{
			%for i in range(max_radix):
				${insertGlobalStore(i, i * block_size, split, normalization_coeff)}
			%endfor
		}
	%endif
</%def>

<%def name="insertfftKernel(radix, num_iter)">
	%for i in range(num_iter):
		fftKernel${radix}<dir>(a + ${i * radix});
	%endfor
</%def>

<%def name="insertTwiddleKernel(radix, num_iter, radix_prev, data_len, threads_per_xform, scalar, complex)">

	<% log2_radix_prev = log2(radix_prev) %>
	{ // Twiddle kernel
		${scalar} angf, ang;
		${complex} w;

	%for z in range(num_iter):
		%if z == 0:
			%if radix_prev > 1:
				angf = (${scalar})(ii >> ${log2_radix_prev});
			%else:
				angf = (${scalar})ii;
			%endif
		%else:
			%if radix_prev > 1:
				angf = (${scalar})((${z * threads_per_xform} + ii) >> ${log2_radix_prev});
			%else:
				## TODO: find out which conditions are necessary to execute this code
				angf = (${scalar})(${z * threads_per_xform} + ii);
			%endif
		%endif

		%for k in range(1, radix):
			<% ind = z * radix + k %>
			ang = dir * (${scalar})${2 * math.pi * k / data_len} * angf;
			complex_exp(w, ang);
			a[${ind}] = a[${ind}] * w;
		%endfor
	%endfor
	}
</%def>

<%def name="insertLocalStores(num_iter, radix, threads_per_xform, threads_req, offset, comp)">
	%for z in range(num_iter):
		%for k in range(radix):
			<% index = k * (threads_req + offset) + z * threads_per_xform %>
			smem[smem_store_index + ${index}] = a[${z * radix + k}].${comp};
		%endfor
	%endfor
	__syncthreads();
</%def>

<%def name="insertLocalLoads(n, radix, radix_next, radix_prev, radix_curr, threads_per_xform, threads_req, offset, comp)">
	<%
		threads_req_next = n / radix_next
		inter_block_hnum = max(radix_prev / threads_per_xform, 1)
		inter_block_hstride = threads_per_xform
		vert_width = max(threads_per_xform / radix_prev, 1)
		vert_width = min(vert_width, radix)
		vert_num = radix / vert_width
		vert_stride = (n / radix + offset) * vert_width
		iter = max(threads_req_next / threads_per_xform, 1)
		intra_block_hstride = max(threads_per_xform / (radix_prev * radix), 1)
		intra_block_hstride *= radix_prev

		stride = threads_req / radix_next
	%>

	%for i in range(iter):
		<%
			ii = i / (inter_block_hnum * vert_num)
			zz = i % (inter_block_hnum * vert_num)
			jj = zz % inter_block_hnum
			kk = zz / inter_block_hnum
		%>

		%for z in range(radix_next):
			<% st = kk * vert_stride + jj * inter_block_hstride + ii * intra_block_hstride + z * stride %>
			a[${i * radix_next + z}].${comp} = smem[smem_load_index + ${st}];
		%endfor
	%endfor
	__syncthreads();
</%def>

<%def name="insertLocalLoadIndexArithmetic(radix_prev, radix, threads_req, threads_per_xform, xforms_per_block, offset, mid_pad)">
	<%
		radix_curr = radix_prev * radix
		log2_radix_curr = log2(radix_curr)
		log2_radix_prev = log2(radix_prev)
		incr = (threads_req + offset) * radix + mid_pad
	%>

	%if radix_curr < threads_per_xform:
		%if radix_prev == 1:
			j = ii & ${radix_curr - 1};
		%else:
			j = (ii & ${radix_curr - 1}) >> ${log2_radix_prev};
		%endif

		%if radix_prev == 1:
			i = ii >> ${log2_radix_curr};
		%else:
			i = mad24(ii >> ${log2_radix_curr}, ${radix_prev}, ii & ${radix_prev - 1});
		%endif
	%else:
		%if radix_prev == 1:
			j = ii;
		%else:
			j = ii >> ${log2_radix_prev};
		%endif

		%if radix_prev == 1:
			i = 0;
		%else:
			i = ii & ${radix_prev - 1};
		%endif
	%endif

	%if xforms_per_block > 1:
		i = mad24(jj, ${incr}, i);
	%endif

	smem_load_index = mad24(j, ${threads_req + offset}, i);
</%def>

<%def name="insertLocalStoreIndexArithmetic(threads_req, xforms_per_block, radix, offset, mid_pad)">
	%if xforms_per_block == 1:
		smem_store_index = ii;
	%else:
		smem_store_index = mad24(jj, ${(threads_req + offset) * radix + mid_pad}, ii);
	%endif
</%def>

<%def name="insertGlobalHeader(name, split, scalar, complex)">
%if split:
	void ${name}(${scalar} *in_real, ${scalar} *in_imag, ${scalar} *out_real, ${scalar} *out_imag, int S)
%else:
	void ${name}(${complex} *in, ${complex} *out, int S)
%endif
</%def>

<%def name="insertKernelTemplateHeader(kernel_name, split, scalar, complex)">
	template<int dir> __device__ ${insertGlobalHeader(kernel_name, split, scalar, complex)}
</%def>

<%def name="insertKernelSpecializations(kernel_name, split, scalar, complex)">
extern "C" {
	%for dir, suffix in ((-1, "_forward"), (1, "_inverse")):
		__global__ ${insertGlobalHeader(kernel_name + suffix, split, scalar, complex)}
		{
		%if split:
			${kernel_name}<${dir}>(in_real, in_imag, out_real, out_imag, S);
		%else:
			${kernel_name}<${dir}>(in, out, S);
		%endif
		}
	%endfor
}
</%def>

<%def name="insertVariableDefinitions(scalar, complex, shared_mem, temp_array_size)">

	%if shared_mem > 0:
		__shared__ ${scalar} smem[${shared_mem}];
		size_t smem_store_index, smem_load_index;
	%endif

	## need to fill a[] with zeros, because otherwise nvcc crashes
	## (it considers a[] not initialized)
	${complex} a[${temp_array_size}] = {${', '.join(['0'] * temp_array_size * 2)}};

	int thread_id = threadIdx.x;
	int block_id = blockIdx.x + blockIdx.y * gridDim.x;
</%def>

<%def name="localKernel(scalar, complex, split, kernel_name, n, radix_arr, shared_mem, \
	threads_per_xform, xforms_per_block, min_mem_coalesce_width, num_smem_banks, normalization_coeff)">

	<%
		max_radix = radix_arr[0]
		radix_prev = 1
		data_len = n
		num_radix = len(radix_arr)
	%>

${insertBaseKernels(scalar, complex)}

${insertKernelTemplateHeader(kernel_name, split, scalar, complex)}
{
	${insertVariableDefinitions(scalar, complex, shared_mem, max_radix)}
	int ii;
	%if num_radix > 1:
		int i, j;
	%endif

	%if not (threads_per_xform >= min_mem_coalesce_width and xforms_per_block == 1):
		int jj, s;
	%endif

	${insertGlobalLoadsAndTranspose(n, threads_per_xform, xforms_per_block, max_radix,
		min_mem_coalesce_width, split)}

	%for r in range(num_radix):
		<%
			num_iter = radix_arr[0] / radix_arr[r]
			threads_req = n / radix_arr[r]
			radix_curr = radix_prev * radix_arr[r]
		%>

		${insertfftKernel(radix_arr[r], num_iter)}

		%if r < num_radix - 1:
			${insertTwiddleKernel(radix_arr[r], num_iter, radix_prev, data_len, threads_per_xform, scalar, complex)}
			<%
				lMemSize, offset, mid_pad = getPadding(threads_per_xform, radix_prev, threads_req,
					xforms_per_block, radix_arr[r], num_smem_banks)
			%>
			${insertLocalStoreIndexArithmetic(threads_req, xforms_per_block, radix_arr[r], offset, mid_pad)}
			${insertLocalLoadIndexArithmetic(radix_prev, radix_arr[r], threads_req, threads_per_xform, xforms_per_block, offset, mid_pad)}
			%for comp in ('x', 'y'):
				${insertLocalStores(num_iter, radix_arr[r], threads_per_xform, threads_req, offset, comp)}
				${insertLocalLoads(n, radix_arr[r], radix_arr[r+1], radix_prev, radix_curr, threads_per_xform, threads_req, offset, comp)}
			%endfor
			<%
				radix_prev = radix_curr
				data_len = data_len / radix_arr[r]
			%>
		%endif
	%endfor

	${insertGlobalStoresAndTranspose(n, max_radix, radix_arr[num_radix - 1], threads_per_xform,
		xforms_per_block, min_mem_coalesce_width, split, normalization_coeff)}
}

${insertKernelSpecializations(kernel_name, split, scalar, complex)}
</%def>

<%def name="globalKernel(scalar, complex, split, kernel_name, n, curr_n, pass_num, shared_mem, \
	batch_size, horiz_bs, vert_bs, vertical, max_block_size, normalization_coeff)">

${insertBaseKernels(scalar, complex)}

	<%
		radix_arr, radix1_arr, radix2_arr = getGlobalRadixInfo(n)

		num_passes = len(radix_arr)

		radix_init = horiz_bs if vertical else 1

		radix = radix_arr[pass_num]
		radix1 = radix1_arr[pass_num]
		radix2 = radix2_arr[pass_num]

		stride_in = radix_init
		for i in range(num_passes):
			if i != pass_num:
				stride_in *= radix_arr[i]

		stride_out = radix_init
		for i in range(pass_num):
			stride_out *= radix_arr[i]

		block_size = min(batch_size * radix2, max_block_size)

		num_iter = radix1 / radix2
		input_multiplier = block_size / batch_size
		log2_stride_out = log2(stride_out)
		blocks_per_xform = stride_in / batch_size

		m = log2(n)
	%>

${insertKernelTemplateHeader(kernel_name, split, scalar, complex)}
{
	${insertVariableDefinitions(scalar, complex, shared_mem, radix1)}
	int index_in, index_out, x_num, tid, i, j;
	%if not vertical or pass_num < num_passes - 1:
		int b_num;
	%endif

	<% log2_blocks_per_xform = log2(blocks_per_xform) %>

	%if vertical:
		x_num = block_id >> ${log2_blocks_per_xform};
		block_id = block_id & ${blocks_per_xform - 1};
		index_in = mad24(block_id, ${batch_size}, x_num << ${log2(n * horiz_bs)});
		tid = mul24(block_id, ${batch_size});
		i = tid >> ${log2_stride_out};
		j = tid & ${stride_out - 1};
		<%
			stride = radix * radix_init
			for i in range(pass_num):
				stride *= radix_arr[i]
		%>
		index_out = mad24(i, ${stride}, j + (x_num << ${log2(n*horiz_bs)}));

		## do not set it, if it won't be used
		%if pass_num < num_passes - 1:
			b_num = block_id;
		%endif
	%else:
		b_num = block_id & ${blocks_per_xform - 1};
		x_num = block_id >> ${log2_blocks_per_xform};
		index_in = mul24(b_num, ${batch_size});
		tid = index_in;
		i = tid >> ${log2_stride_out};
		j = tid & ${stride_out - 1};
		<%
			stride = radix*radix_init
			for i in range(pass_num):
				stride *= radix_arr[i]
		%>
		index_out = mad24(i, ${stride}, j);
		index_in += (x_num << ${m});
		index_out += (x_num << ${m});
	%endif

	## Load Data
	<% log2_batch_size = log2(batch_size) %>
	tid = thread_id;
	i = tid & ${batch_size - 1};
	j = tid >> ${log2_batch_size};
	index_in += mad24(j, ${stride_in}, i);

	%if split:
		in_real += index_in;
		in_imag += index_in;
		%for comp, part in (('x', 'real'), ('y', 'imag')):
			%for j in range(radix1):
				a[${j}].${comp} = in_${part}[${j * input_multiplier * stride_in}];
			%endfor
		%endfor
	%else:
		in += index_in;
		%for j in range(radix1):
			a[${j}] = in[${j * input_multiplier * stride_in}];
		%endfor
	%endif

	fftKernel${radix1}<dir>(a);

	%if radix2 > 1:
		## twiddle
		{
			${scalar} ang;
			${complex} w;

		%for k in range(1, radix1):
			ang = dir * (${scalar})${2 * math.pi * k / radix} * j;
			complex_exp(w, ang);
			a[${k}] = a[${k}] * w;
		%endfor
		}

		## shuffle
		index_in = mad24(j, ${block_size * num_iter}, i);
		smem_store_index = tid;
		smem_load_index = index_in;

		%for comp in ('x', 'y'):
			%for k in range(radix1):
				smem[smem_store_index + ${k * block_size}] = a[${k}].${comp};
			%endfor
			__syncthreads();

			%for k in range(num_iter):
				%for t in range(radix2):
					a[${k * radix2 + t}].${comp} = smem[smem_load_index + ${t * batch_size + k * block_size}];
				%endfor
			%endfor
			__syncthreads();
		%endfor

		%for j in range(num_iter):
			fftKernel${radix2}<dir>(a + ${j * radix2});
		%endfor
	%endif

	## twiddle
	%if pass_num < num_passes - 1:
	{
		${scalar} ang1, ang;
		${complex} w;

		int l = ((b_num << ${log2_batch_size}) + i) >> ${log2_stride_out};
		int k = j << ${log2(radix1 / radix2)};
		ang1 = dir * (${scalar})${2 * math.pi / curr_n} * l;
		%for t in range(radix1):
			ang = ang1 * (k + ${(t % radix2) * radix1 + (t / radix2)});
			complex_exp(w, ang);
			a[${t}] = a[${t}] * w;
		%endfor
	}
	%endif

	## Store Data
	%if stride_out == 1:
		smem_store_index = mad24(i, ${radix + 1}, j << ${log2(radix1 / radix2)});
		smem_load_index = mad24(tid >> ${log2(radix)}, ${radix + 1}, tid & ${radix - 1});

		%for comp in ('x', 'y'):
			%for i in range(radix1 / radix2):
				%for j in range(radix2):
					smem[smem_store_index + ${i + j * radix1}] = a[${i * radix2 + j}].${comp};
				%endfor
			%endfor
			__syncthreads();

			%for i in range(radix1):
				a[${i}].${comp} = smem[smem_load_index + ${i * (radix + 1) * (block_size / radix)}];
			%endfor
			__syncthreads();
		%endfor

		index_out += tid;

		%if split:
			out_real += index_out;
			out_imag += index_out;

			%for comp, part in (('x', 'real'), ('y', 'imag')):
				%for k in range(radix1):
					out_${part}[${k * block_size}] = a[${k}].${comp};
				%endfor
			%endfor
		%else:
			out += index_out;
			%for k in range(radix1):
				out[${k * block_size}] = a[${k}];
			%endfor
		%endif
	%else:
		index_out += mad24(j, ${num_iter * stride_out}, i);
		%if split:
			out_real += index_out;
			out_imag += index_out;
			%for comp, part in (('x', 'real'), ('y', 'imag')):
				%for k in range(radix1):
					out_${part}[${((k % radix2) * radix1 + (k / radix2)) * stride_out}] =
						a[${k}].${comp} / (dir == 1 ? ${normalization_coeff} : 1);
				%endfor
			%endfor
		%else:
			out += index_out;
			%for k in range(radix1):
				out[${((k % radix2) * radix1 + (k / radix2)) * stride_out}] =
					a[${k}] / (dir == 1 ? ${normalization_coeff} : 1);
			%endfor
		%endif
	%endif
}

${insertKernelSpecializations(kernel_name, split, scalar, complex)}
</%def>
