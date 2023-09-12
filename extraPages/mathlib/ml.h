#pragma once
#include <Eigen/Core>
#include <immintrin.h>
#include <type_traits>

#define noinline_ __attribute__((noinline))

	typedef double __m256d_x4 __attribute__ ((__vector_size__ (32*4),
						__may_alias__));

	// using Vec4 = __m256d;
	// using Mat4 = __m256d_x4;
	// using Vec4 = alignas(32) double*;
	// using Mat4 = alignas(32) double*;
/*
struct Vec4 {
	alignas(32) double d[4];

	inline Vec4(const double* u) {
		for (int i=0; i<4; i++) d[i] = u[i];
	}

	inline Vec4(const _mm256d& v) {
		_mm256_store_pd(d,v);
	}

	inline __m256d pack() {
		return _mm256_load_pd(d);
	}
};
struct Mat4 {
	alignas(32) double d[16];
};
*/

template <int N>
struct Array {
	inline static constexpr int SN = ((N+3)/4)*4;
	alignas(32) double d[SN];

	// Empty.
	inline Array() {
	}

	// Build from unaligned array.
	inline Array(const double* u) {
		for (int i=0; i<N;  i++) d[i] = u[i];
		for (int i=N; i<SN; i++) d[i] = 0;
	}

	// Build from multiple vector parts.
	inline Array(const __m256d& a) {
		set_(a, 0);
	}

	// Build from multiple vector parts.
	template <class... Args>
	inline Array(const __m256d& a, const Args&... args) {
		int i = 0;
		set_(a, i++);
		for (const __m256d& arg : {args...}) {
			set_(arg, i++);
		}
	}

	// Set the i'th vector part.
	// Always treat as quad-double, then it is always aligned,
	// so store can be used rather than storeu.
	inline void set_(const __m256d& v, int i) {
		_mm256_store_pd(d+i*4, v);
	}

	inline const double& operator[](int i) const { return d[i]; }
	inline double& operator[](int i) { return d[i]; }

	inline __m256d packetv(int i) const {
		return _mm256_load_pd(d+4*i);
	}

	/*
	inline __m256d rowv(int r) const {
		// return _mm256_load_pd(d+4*r);
		return _mm256_set_pd(d[r*4+3], d[r*4+2], d[r*4+1], d[r*4+0]);
	}
	inline __m256d colv(int r) const {
		// return _mm256_set_pd(d[0*4+r], d[1*4+r], d[2*4+r], d[3*4+r]);
		return _mm256_set_pd(d[3*4+r], d[2*4+r], d[1*4+r], d[0*4+r]);
	}
	*/
	inline __m256d blockv(int r, int c, int C) const {
		return _mm256_load_pd(d+r*C*4+c*4);
	}
	inline __m256d blockv_t(int r, int c, int R) const {
		// return _mm256_set_pd(d[0*4*R+4*r], d[1*4*R+4*r], d[2*4*R+4*r], d[3*4*R+4*r]);
		// return _mm256_set_pd(d[(0+c)*R+r], d[(1+c)*R+r], d[(2+c)*R+r], d[(3+c)*R+r]);
		// return _mm256_set_pd(d[(0+r)*R+c], d[(1+r)*R+c], d[(2+r)*R+c], d[(3+r)*R+c]);
		return _mm256_set_pd(d[(3+r)*R+c], d[(2+r)*R+c], d[(1+r)*R+c], d[(0+r)*R+c]);
	}

};

using Vec4 = Array<4>;
using Mat4 = Array<16>;
using Vec20 = Array<20>;
using Mat20 = Array<20*20>;

	void f_vec4(Vec4& v);
	void f_ptr(double* d);

	Mat4 hello();


	Mat4 transpose(const Mat4& A);
	Vec4 vec4(const double* vs);
	Mat4 mat4(const double* vs);
	// Vec4 mv_loop(const double* A, const double* x);
	// Vec4 mv_loop(const Mat4& A, const Vec4& x);
	// Vec4 mv_hadd(const Mat4& A, const Vec4& x);
	// Vec4 mv_transpose_explicit(const Mat4& A, const Vec4& x);
	// Vec4 mv_transpose_implicit(const Mat4& A, const Vec4& x);
	void print(const __m256d& p);
	void print(const double* A, int m, int n);
	void print(const Vec4& a);
	void print(const Mat4& A);

	template <int N>
	Array<N> mv_transpose_implicit(const Array<N*N>& A, const Array<N>& x);
	template <int N>
	Array<N> mv_loop(const Array<N*N>& A, const Array<N>& x);





// namespace e {
	// using namespace Eigen;
	// using Vec4 = Eigen::Vector4d;
	// using Mat4 = Eigen::Matrix<double,4,4,Eigen::RowMajor>;

	template <int N>
	// Array<N> mv_eigen(const Map<Array<N*N>>& A, const Map<Array<N>>& x);
	// Array<N> mv_eigen(const Eigen::Map<Eigen::Matrix<double,N,N,Eigen::RowMajor>>& A, const Eigen::Map<Eigen::Matrix<double,N,1>>& x);
	Array<N> mv_eigen(const Array<N*N>& A, const Array<N>& x);
// }
