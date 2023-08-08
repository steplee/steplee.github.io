#include "ml.h"
#include <iostream>

	void noinline_ f_vec4(Vec4& v) {}
	void noinline_ f_ptr(double* d) {}

	Mat4 hello() {
		auto a = _mm256_set1_pd(1);
		auto b = _mm256_set1_pd(2);
		return Array<16>(a,b,a,b);
	}


// namespace e{
	template <int N>
	// Array<N> mv_eigen(const Eigen::Map<Eigen::Matrix<double,N,N,Eigen::RowMajor>>& A, const Eigen::Map<Eigen::Matrix<double,N,1>>& x) {
	Array<N> mv_eigen(const Array<N*N>& AA, const Array<N>& xx) {
		Array<N> out;
		Eigen::Map<Eigen::Matrix<double,N,1>> outView(out.d);
		Eigen::Map<const Eigen::Matrix<double,N,N,Eigen::RowMajor>> A(AA.d);
		Eigen::Map<const Eigen::Matrix<double,N,1>> x(xx.d);
		outView = A*x;
		return out;
	}

	template Array<4> mv_eigen(const Array<4*4>& AA, const Array<4>& xx);
	template Array<8> mv_eigen(const Array<8*8>& AA, const Array<8>& xx);
	template Array<20> mv_eigen(const Array<20*20>& AA, const Array<20>& xx);
// }



	Mat4 transpose(const Mat4& A) {
		Mat4 o;
		for (int i=0; i<4; i++)
		for (int j=0; j<4; j++) {
			o.d[i*4+j] = A.d[j*4+i];
		}
		return o;
	}

	Vec4 vec4(const double* vs) {
		return Vec4(vs);
	}

	Mat4 mat4(const double* vs) {
		return Mat4(vs);
	}

/*
	Vec4 mv_loop(const Mat4& A, const Vec4& u) {
		Vec4 o;
		// o[0] = A[0*4+0] * u[0] + A[0*4+1] * u[1] + A[0*4+2] * u[2] + A[0*4+3] * u[3];
		// o[1] = A[1*4+0] * u[0] + A[1*4+1] * u[1] + A[1*4+2] * u[2] + A[1*4+3] * u[3];
		// o[2] = A[2*4+0] * u[0] + A[2*4+1] * u[1] + A[2*4+2] * u[2] + A[2*4+3] * u[3];
		// o[3] = A[3*4+0] * u[0] + A[3*4+1] * u[1] + A[3*4+2] * u[2] + A[3*4+3] * u[3];
		for (int i=0; i<4; i++) {
			double s = 0;
			for (int j=0; j<4; j++) {
				s += A[i*4+j] * u[j];
			}
			o[i] = s;
		}
		return o;
	}

	Vec4 mv_hadd(const Mat4& A, const Vec4& x) {
		// This is slower than the other. I suspect extract/insert/cast is slow.
		// Apparently the avx registers are implemented like 2x2, and swizzling across
		// the half boundary is not great.
		auto xx = x.rowv(0);
		auto d1 = _mm256_mul_pd(A.rowv(0), xx);
		auto d2 = _mm256_mul_pd(A.rowv(1), xx);
		auto s12  = _mm256_hadd_pd(d1,d2);
		auto t12  = _mm256_extractf128_pd(s12, 1);
		auto u12  = _mm_add_pd(t12, _mm256_castpd256_pd128(s12)); // 128.

		auto d3 = _mm256_mul_pd(A.rowv(2), xx);
		auto d4 = _mm256_mul_pd(A.rowv(3), xx);
		auto s34  = _mm256_hadd_pd(d3,d4);
		auto t34  = _mm256_extractf128_pd(s34, 1);
		auto u34  = _mm_add_pd(t34, _mm256_castpd256_pd128(s34)); // 128.

		auto lo   = _mm256_castpd128_pd256(u12);
		auto both = _mm256_insertf128_pd(lo,u34,1);
		Vec4 o;
		_mm256_store_pd(o.d, both);
		return o;
	}

	Vec4 mv_transpose_explicit(const Mat4& A, const Vec4& x) {
		auto At = transpose(A);
		__m256d d,o;

		o = _mm256_mul_pd(_mm256_set1_pd(x[0]), At.rowv(0));
		d = _mm256_mul_pd(_mm256_set1_pd(x[1]), At.rowv(1));
		o = _mm256_add_pd(o, d);
		d = _mm256_mul_pd(_mm256_set1_pd(x[2]), At.rowv(2));
		o = _mm256_add_pd(o, d);
		d = _mm256_mul_pd(_mm256_set1_pd(x[3]), At.rowv(3));
		o = _mm256_add_pd(o, d);
		return Vec4{o};
	}

	static Vec4 mv_transpose_implicit_(const double* __restrict__ A, const double* __restrict__ x) {
		__m256d d,o;
		o = _mm256_mul_pd(_mm256_set1_pd(x[0]), ((Mat4*)A)->colv(0));
		d = _mm256_mul_pd(_mm256_set1_pd(x[1]), ((Mat4*)A)->colv(1));
		o = _mm256_add_pd(o, d);
		d = _mm256_mul_pd(_mm256_set1_pd(x[2]), ((Mat4*)A)->colv(2));
		o = _mm256_add_pd(o, d);
		d = _mm256_mul_pd(_mm256_set1_pd(x[3]), ((Mat4*)A)->colv(3));
		o = _mm256_add_pd(o, d);
		return Vec4{o};
	}
	Vec4 mv_transpose_implicit(const Mat4& A, const Vec4& x) {
		// return mv_transpose_implicit_(&A[0], &x[0]);
		__m256d d,o;
		o = _mm256_mul_pd(_mm256_set1_pd(x[0]), A.colv(0));
		d = _mm256_mul_pd(_mm256_set1_pd(x[1]), A.colv(1));
		o = _mm256_add_pd(o, d);
		d = _mm256_mul_pd(_mm256_set1_pd(x[2]), A.colv(2));
		o = _mm256_add_pd(o, d);
		d = _mm256_mul_pd(_mm256_set1_pd(x[3]), A.colv(3));
		o = _mm256_add_pd(o, d);
		return Vec4{o};
	}
*/

	template <int N>
	Array<N> mv_transpose_implicit(const Array<N*N>& A, const Array<N>& x) {
		Array<N> o;
		__m256d d;
		alignas(32) double z_[4]={0,0,0,0};
		auto z4 = Vec4{z_}.packetv(0);
		for (int i=0; i<N/4; i++) {
			auto o4 = z4;
			// for (int j=0; j<N/4; j++) {
			for (int j=0; j<N; j++) {
				// auto xx = x.packetv(i);
				auto xx = _mm256_set1_pd(x[j]);
				auto t = _mm256_mul_pd(A.blockv_t(i*4,j,N), xx);
				// std::cout << " - a  : "; print(A.blockv_t(i*4,j,N)); std::cout << "\n";
				// std::cout << " - b  : "; print(xx); std::cout << "\n";
				// std::cout << " - dot: "; print(t); std::cout << "\n\n";
				o4 = _mm256_add_pd(o4, t);
			}
			_mm256_store_pd(o.d+i*4, o4);
		}
		return o;
	}

	template <int N>
	Array<N> mv_loop(const Array<N*N>& A, const Array<N>& x) {
		Array<N> o;
		for (int i=0; i<N; i++) {
			double s = 0;
			for (int j=0; j<N; j++) {
				s += A[i*N+j] * x[j];
			}
			o[i] = s;
		}
		return o;
	}

// Explicit instant
constexpr static int n0=4;
template Array<n0> mv_transpose_implicit(const Array<n0*n0>&, const Array<n0>&);
template Array<n0> mv_loop(const Array<n0*n0>&, const Array<n0>&);

constexpr static int n1=20;
template Array<n1> mv_transpose_implicit(const Array<n1*n1>&, const Array<n1>&);
template Array<n1> mv_loop(const Array<n1*n1>&, const Array<n1>&);

constexpr static int n2=8;
template Array<n2> mv_transpose_implicit(const Array<n2*n2>&, const Array<n2>&);
template Array<n2> mv_loop(const Array<n2*n2>&, const Array<n2>&);


	void print(const __m256d& p) {
		alignas(32) double t[4];
		_mm256_store_pd(t,p);
		std::cout << t[0] << ", " << t[1] << ", " << t[2] << ", " << t[3];
	}
	void print(const double* A, int m, int n) {
		for (int i=0; i<m; i++) {
			for (int j=0; j<n; j++) {
				std::cout << A[i*n+j];
				if (j<n-1) std::cout << ", ";
				else std::cout << "\n";
			}
		}
	}
	void print(const Vec4& a) {
		std::cout << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3];
	}
	void print(const Mat4& A) {
		print(A.d,4,4);
	}

#if 0
	namespace {
		__m256d* rowp(const Mat4& A, int r) {
			return (__m256d*)(((const double*)&A) + 4*r);
		}
		__m256d row(const Mat4& A, int r) {
			return *(__m256d*)(((const double*)&A) + 4*r);
		}
	}

	Mat4 transpose(const Mat4& A) {
		Mat4 o;
		alignas(32) double t[16];
		_mm256_store_pd(t+0, *rowp(A,0));
		_mm256_store_pd(t+4, *rowp(A,1));
		_mm256_store_pd(t+8, *rowp(A,2));
		_mm256_store_pd(t+12, *rowp(A,3));
		// Vector registers are actually reversed.
		*rowp(o,3) = _mm256_set_pd(t[3*4+3], t[2*4+3], t[1*4+3], t[0*4+3]);
		*rowp(o,2) = _mm256_set_pd(t[3*4+2], t[2*4+2], t[1*4+2], t[0*4+2]);
		*rowp(o,1) = _mm256_set_pd(t[3*4+1], t[2*4+1], t[1*4+1], t[0*4+1]);
		*rowp(o,0) = _mm256_set_pd(t[3*4+0], t[2*4+0], t[1*4+0], t[0*4+0]);
		return o;
	}

	Vec4 vec4(const double* vs) {
		// return _mm256_load_pd(vs[0],vs[1],vs[2],vs[3]);
		// return _mm256_load_pd(vs);
		return _mm256_loadu_pd(vs);
	}

	Mat4 mat4(const double* vs) {
		// return _mm256_load_pd(vs[0],vs[1],vs[2],vs[3]);
		// return _mm256_load_pd(vs);
		Mat4 o;
		*rowp(o,0) = vec4(vs+0);
		*rowp(o,1) = vec4(vs+4);
		*rowp(o,2) = vec4(vs+8);
		*rowp(o,3) = vec4(vs+12);
		return o;
	}

/*
	Vec4 mv_loop(const double* A, const double* x) {
		alignas(32) double o[4];
		for (int i=0; i<4; i++) {
			double s = 0;
			for (int j=0; j<4; j++) {
				s += A[i*4+j] * x[j];
			}
			o[i] = s;
		}
		return _mm256_load_pd(o);
	}


	Vec4 noinline_ mv_hadd(const Mat4& A, const Vec4& x) {
		auto d1 = _mm256_mul_pd(*rowp(A,0), x);
		auto d2 = _mm256_mul_pd(*rowp(A,1), x);
		auto s12  = _mm256_hadd_pd(d1,d2);
		auto t12  = _mm256_extractf128_pd(s12, 1);
		auto u12  = _mm_add_pd(t12, _mm256_castpd256_pd128(s12)); // 128.

		auto d3 = _mm256_mul_pd(*rowp(A,2), x);
		auto d4 = _mm256_mul_pd(*rowp(A,3), x);
		auto s34  = _mm256_hadd_pd(d3,d4);
		auto t34  = _mm256_extractf128_pd(s34, 1);
		auto u34  = _mm_add_pd(t34, _mm256_castpd256_pd128(s34)); // 128.

		auto lo = _mm256_castpd128_pd256(u12);
		Vec4 o  = _mm256_insertf128_pd(lo,u34,1);
		return o;
	}

	Vec4 noinline_ mv_transpose(const Mat4& A, const Vec4& x) {
		auto At = transpose(A);

		__m256d d;
		alignas(32) double xx[4];
		_mm256_store_pd(xx,x);

		Vec4 o;
		o = _mm256_mul_pd(_mm256_set1_pd(xx[0]), *rowp(At,0));
		d = _mm256_mul_pd(_mm256_set1_pd(xx[1]), *rowp(At,1));
		o = _mm256_add_pd(o, d);
		d = _mm256_mul_pd(_mm256_set1_pd(xx[2]), *rowp(At,2));
		o = _mm256_add_pd(o, d);
		d = _mm256_mul_pd(_mm256_set1_pd(xx[3]), *rowp(At,3));
		o = _mm256_add_pd(o, d);
		return o;

	}
*/

	void print(const double* A, int m, int n) {
		for (int i=0; i<m; i++) {
			for (int j=0; j<n; j++) {
				std::cout << A[i*n+j];
				if (j<n-1) std::cout << ", ";
				else std::cout << "\n";
			}
		}
	}
	void print(const Vec4& a) {
		// __attribute__((aligned)) double t[4];
		alignas(32) double t[32];
		_mm256_store_pd(t, a);
		std::cout << t[0] << ", " << t[1] << ", " << t[2] << ", " << t[3];
	}
	void print(const Mat4& A) {
		alignas(32) double t[16];
		_mm256_store_pd(t+0, *rowp(A,0));
		_mm256_store_pd(t+4, *rowp(A,1));
		_mm256_store_pd(t+8, *rowp(A,2));
		_mm256_store_pd(t+12, *rowp(A,3));
		print(t,4,4);
	}
#endif
