from mako.template import Template

base_kernels = Template("""
#ifndef M_PI
#define M_PI 0x1.921fb54442d18p+1
#endif

inline float2 operator*( float2 a, float2 b ) { return make_float2( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }
inline float2 operator*( float2 a, float  b ) { return make_float2( b*a.x, b*a.y ); }
inline float2 operator+( float2 a, float2 b ) { return make_float2( a.x + b.x, a.y + b.y ); }
inline float2 operator-( float2 a, float2 b ) { return make_float2( a.x - b.x, a.y - b.y ); }

// TODO: 'mad' seems to be some addition + multiplication function
//#define complexMul(a,b) (make_float2(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y)))
#define mad24(x, y, z) ((x) * (y) + (z))
#define complexMul(a, b) ((a) * (b))

#define conj(a) make_float2((a).x, -(a).y)
#define conjTransp(a) make_float2(-(a).y, (a).x)

#define fftKernel2(a, dir) \\\\\n
{ \\\\\n
	float2 c = (a)[0]; \\\\\n
	(a)[0] = c + (a)[1]; \\\\\n
	(a)[1] = c - (a)[1]; \\\\\n
}

#define fftKernel2S(d1, d2, dir) \\\\\n
{ \\\\\n
	float2 c = (d1); \\\\\n
	(d1) = c + (d2); \\\\\n
	(d2) = c - (d2); \\\\\n
}

#define fftKernel4(a, dir) \\\\\n
{ \\\\\n
	fftKernel2S((a)[0], (a)[2], dir); \\\\\n
	fftKernel2S((a)[1], (a)[3], dir); \\\\\n
	fftKernel2S((a)[0], (a)[1], dir); \\\\\n
	(a)[3] = conjTransp((a)[3]) * dir; \\\\\n
	fftKernel2S((a)[2], (a)[3], dir); \\\\\n
	float2 c = (a)[1]; \\\\\n
	(a)[1] = (a)[2]; \\\\\n
	(a)[2] = c; \\\\\n
}

#define fftKernel4s(a0, a1, a2, a3, dir) \\\\\n
{ \\\\\n
	fftKernel2S((a0), (a2), dir); \\\\\n
	fftKernel2S((a1), (a3), dir); \\\\\n
	fftKernel2S((a0), (a1), dir); \\\\\n
	(a3) = conjTransp((a3)) * dir; \\\\\n
	fftKernel2S((a2), (a3), dir); \\\\\n
	float2 c = (a1); \\\\\n
	(a1) = (a2); \\\\\n
	(a2) = c; \\\\\n
}

#define bitreverse8(a) \\\\\n
{ \\\\\n
	float2 c; \\\\\n
	c = (a)[1]; \\\\\n
	(a)[1] = (a)[4]; \\\\\n
	(a)[4] = c; \\\\\n
	c = (a)[3]; \\\\\n
	(a)[3] = (a)[6]; \\\\\n
	(a)[6] = c; \\\\\n
}

#define fftKernel8(a, dir) \\\\\n
{ \\\\\n
	const float2 w1  = make_float2(0x1.6a09e6p-1f,  dir*0x1.6a09e6p-1f); \\\\\n
	const float2 w3  = make_float2(-0x1.6a09e6p-1f, dir*0x1.6a09e6p-1f); \\\\\n
	fftKernel2S((a)[0], (a)[4], dir); \\\\\n
	fftKernel2S((a)[1], (a)[5], dir); \\\\\n
	fftKernel2S((a)[2], (a)[6], dir); \\\\\n
	fftKernel2S((a)[3], (a)[7], dir); \\\\\n
	(a)[5] = complexMul(w1, (a)[5]); \\\\\n
	(a)[6] = conjTransp((a)[6]) * dir; \\\\\n
	(a)[7] = complexMul(w3, (a)[7]); \\\\\n
	fftKernel2S((a)[0], (a)[2], dir); \\\\\n
	fftKernel2S((a)[1], (a)[3], dir); \\\\\n
	fftKernel2S((a)[4], (a)[6], dir); \\\\\n
	fftKernel2S((a)[5], (a)[7], dir); \\\\\n
	(a)[3] = conjTransp((a)[3]) * dir; \\\\\n
	(a)[7] = conjTransp((a)[7]) * dir; \\\\\n
	fftKernel2S((a)[0], (a)[1], dir); \\\\\n
	fftKernel2S((a)[2], (a)[3], dir); \\\\\n
	fftKernel2S((a)[4], (a)[5], dir); \\\\\n
	fftKernel2S((a)[6], (a)[7], dir); \\\\\n
	bitreverse8((a)); \\\\\n
}

#define bitreverse4x4(a) \\\\\n
{ \\\\\n
	float2 c; \\\\\n
	c = (a)[1];  (a)[1]  = (a)[4];  (a)[4]  = c; \\\\\n
	c = (a)[2];  (a)[2]  = (a)[8];  (a)[8]  = c; \\\\\n
	c = (a)[3];  (a)[3]  = (a)[12]; (a)[12] = c; \\\\\n
	c = (a)[6];  (a)[6]  = (a)[9];  (a)[9]  = c; \\\\\n
	c = (a)[7];  (a)[7]  = (a)[13]; (a)[13] = c; \\\\\n
	c = (a)[11]; (a)[11] = (a)[14]; (a)[14] = c; \\\\\n
}

#define fftKernel16(a, dir) \\\\\n
{ \\\\\n
	const float w0 = 0x1.d906bcp-1f; \\\\\n
	const float w1 = 0x1.87de2ap-2f; \\\\\n
	const float w2 = 0x1.6a09e6p-1f; \\\\\n
	fftKernel4s((a)[0], (a)[4], (a)[8],  (a)[12], dir); \\\\\n
	fftKernel4s((a)[1], (a)[5], (a)[9],  (a)[13], dir); \\\\\n
	fftKernel4s((a)[2], (a)[6], (a)[10], (a)[14], dir); \\\\\n
	fftKernel4s((a)[3], (a)[7], (a)[11], (a)[15], dir); \\\\\n
	(a)[5]  = complexMul((a)[5], make_float2(w0, dir*w1)); \\\\\n
	(a)[6]  = complexMul((a)[6], make_float2(w2, dir*w2)); \\\\\n
	(a)[7]  = complexMul((a)[7], make_float2(w1, dir*w0)); \\\\\n
	(a)[9]  = complexMul((a)[9], make_float2(w2, dir*w2)); \\\\\n
	(a)[10] = make_float2(dir, 0)*(conjTransp((a)[10])); \\\\\n
	(a)[11] = complexMul((a)[11], make_float2(-w2, dir*w2)); \\\\\n
	(a)[13] = complexMul((a)[13], make_float2(w1, dir*w0)); \\\\\n
	(a)[14] = complexMul((a)[14], make_float2(-w2, dir*w2)); \\\\\n
	(a)[15] = complexMul((a)[15], make_float2(-w0, dir*-w1)); \\\\\n
	fftKernel4((a), dir); \\\\\n
	fftKernel4((a) + 4, dir); \\\\\n
	fftKernel4((a) + 8, dir); \\\\\n
	fftKernel4((a) + 12, dir); \\\\\n
	bitreverse4x4((a)); \\\\\n
}

#define bitreverse32(a) \\\\\n
{ \\\\\n
	float2 c1, c2; \\\\\n
	c1 = (a)[2];   (a)[2] = (a)[1];   c2 = (a)[4];   (a)[4] = c1;   c1 = (a)[8]; \\\\\n
	(a)[8] = c2;    c2 = (a)[16];  (a)[16] = c1;   (a)[1] = c2; \\\\\n
	c1 = (a)[6];   (a)[6] = (a)[3];   c2 = (a)[12];  (a)[12] = c1;  c1 = (a)[24]; \\\\\n
	(a)[24] = c2;   c2 = (a)[17];  (a)[17] = c1;   (a)[3] = c2; \\\\\n
	c1 = (a)[10];  (a)[10] = (a)[5];  c2 = (a)[20];  (a)[20] = c1;  c1 = (a)[9]; \\\\\n
	(a)[9] = c2;    c2 = (a)[18];  (a)[18] = c1;   (a)[5] = c2; \\\\\n
	c1 = (a)[14];  (a)[14] = (a)[7];  c2 = (a)[28];  (a)[28] = c1;  c1 = (a)[25]; \\\\\n
	(a)[25] = c2;   c2 = (a)[19];  (a)[19] = c1;   (a)[7] = c2; \\\\\n
	c1 = (a)[22];  (a)[22] = (a)[11]; c2 = (a)[13];  (a)[13] = c1;  c1 = (a)[26]; \\\\\n
	(a)[26] = c2;   c2 = (a)[21];  (a)[21] = c1;   (a)[11] = c2; \\\\\n
	c1 = (a)[30];  (a)[30] = (a)[15]; c2 = (a)[29];  (a)[29] = c1;  c1 = (a)[27]; \\\\\n
	(a)[27] = c2;   c2 = (a)[23];  (a)[23] = c1;   (a)[15] = c2; \\\\\n
}

#define fftKernel32(a, dir) \\\\\n
{ \\\\\n
	fftKernel2S((a)[0],  (a)[16], dir); \\\\\n
	fftKernel2S((a)[1],  (a)[17], dir); \\\\\n
	fftKernel2S((a)[2],  (a)[18], dir); \\\\\n
	fftKernel2S((a)[3],  (a)[19], dir); \\\\\n
	fftKernel2S((a)[4],  (a)[20], dir); \\\\\n
	fftKernel2S((a)[5],  (a)[21], dir); \\\\\n
	fftKernel2S((a)[6],  (a)[22], dir); \\\\\n
	fftKernel2S((a)[7],  (a)[23], dir); \\\\\n
	fftKernel2S((a)[8],  (a)[24], dir); \\\\\n
	fftKernel2S((a)[9],  (a)[25], dir); \\\\\n
	fftKernel2S((a)[10], (a)[26], dir); \\\\\n
	fftKernel2S((a)[11], (a)[27], dir); \\\\\n
	fftKernel2S((a)[12], (a)[28], dir); \\\\\n
	fftKernel2S((a)[13], (a)[29], dir); \\\\\n
	fftKernel2S((a)[14], (a)[30], dir); \\\\\n
	fftKernel2S((a)[15], (a)[31], dir); \\\\\n
	(a)[17] = complexMul((a)[17], make_float2(0x1.f6297cp-1f, dir*0x1.8f8b84p-3f)); \\\\\n
	(a)[18] = complexMul((a)[18], make_float2(0x1.d906bcp-1f, dir*0x1.87de2ap-2f)); \\\\\n
	(a)[19] = complexMul((a)[19], make_float2(0x1.a9b662p-1f, dir*0x1.1c73b4p-1f)); \\\\\n
	(a)[20] = complexMul((a)[20], make_float2(0x1.6a09e6p-1f, dir*0x1.6a09e6p-1f)); \\\\\n
	(a)[21] = complexMul((a)[21], make_float2(0x1.1c73b4p-1f, dir*0x1.a9b662p-1f)); \\\\\n
	(a)[22] = complexMul((a)[22], make_float2(0x1.87de2ap-2f, dir*0x1.d906bcp-1f)); \\\\\n
	(a)[23] = complexMul((a)[23], make_float2(0x1.8f8b84p-3f, dir*0x1.f6297cp-1f)); \\\\\n
	(a)[24] = complexMul((a)[24], make_float2(0x0p+0f, dir*0x1p+0f)); \\\\\n
	(a)[25] = complexMul((a)[25], make_float2(-0x1.8f8b84p-3f, dir*0x1.f6297cp-1f)); \\\\\n
	(a)[26] = complexMul((a)[26], make_float2(-0x1.87de2ap-2f, dir*0x1.d906bcp-1f)); \\\\\n
	(a)[27] = complexMul((a)[27], make_float2(-0x1.1c73b4p-1f, dir*0x1.a9b662p-1f)); \\\\\n
	(a)[28] = complexMul((a)[28], make_float2(-0x1.6a09e6p-1f, dir*0x1.6a09e6p-1f)); \\\\\n
	(a)[29] = complexMul((a)[29], make_float2(-0x1.a9b662p-1f, dir*0x1.1c73b4p-1f)); \\\\\n
	(a)[30] = complexMul((a)[30], make_float2(-0x1.d906bcp-1f, dir*0x1.87de2ap-2f)); \\\\\n
	(a)[31] = complexMul((a)[31], make_float2(-0x1.f6297cp-1f, dir*0x1.8f8b84p-3f)); \\\\\n
	fftKernel16((a), dir); \\\\\n
	fftKernel16((a) + 16, dir); \\\\\n
	bitreverse32((a)); \\\\\n
}
""")
