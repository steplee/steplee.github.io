
#include "ml.h"
#include <iostream>
#include <iomanip>

#include <benchmark/benchmark.h>

void f_vec4(Vec4& v);
void f_ptr(double* v);

template <int N>
static std::pair<Array<N*N>, Array<N>> get_example() {
	double vv[N];
	double AA[N*N];
	for (int i=0; i<N*N; i++) AA[i] = i/100.;
	for (int i=0; i<N; i++) vv[i] = i/100.;
	auto v = Array<N>(vv);
	auto A = Array<N*N>(AA);
	return {A,v};
}

template <int N>
bool equalMat(const Array<N*N>& A, const Array<N*N>& B) {
	bool equal = true;
	for (int i=0; i<N*N; i++) equal &= (A[i] == B[i]);



	if (not equal) {
		std::cout << std::setprecision(10);
		// Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
		// std::cout << std::setprecision(10);

		std::cout << " - Expected equality between:\n";
		print(A.d,N,N);
		std::cout << " - and:\n";
		print(B.d,N,N);
		assert(false);
	}
	return equal;
}

template <int N>
bool equalVec(const Array<N>& A, const Array<N>& B) {
	bool equal = true;
	// for (int i=0; i<N; i++) equal &= (A[i] == B[i]);
	for (int i=0; i<N; i++) equal &= std::abs(A[i] - B[i]) < 1e-10;
	if (not equal) {
		std::cout << std::setprecision(10);
		std::cout << " - Expected equality between:\n";
		print(A.d,1,N);
		std::cout << " - and:\n";
		print(B.d,1,N);
		for (int i=0; i<N; i++) std::cout << B[i]-A[i] << " "; std::cout << "\n";
		assert(false);
	}
	return equal;
}

static void tests(benchmark::State& state) {
	auto h = hello();

	constexpr int N = 20;
	auto tmp = get_example<N>(); auto A = tmp.first; auto v = tmp.second;
	auto gt = mv_eigen(A,v);

	std::cout << " - test problem, A:\n"; print(A.d,N,N);
	std::cout << " - test problem, u: "; print(v.d,1,N);

	{
		auto tmp = get_example<N>(); auto A = tmp.first; auto v = tmp.second;

		// equalVec<N>(gt, mv_hadd(A,v));
		equalVec<N>(gt, mv_transpose_implicit<N>(A,v));
	}

	// std::cout << " - hello:\n"; print(h); std::cout << std::endl;
}


/*
static void bm_mv_hadd(benchmark::State& state) {
	auto tmp = get_example<4>(); auto A = tmp.first; auto v = tmp.second;

	for (auto _ : state) {
		v.d[0] += 1e-5;
		auto y = mv_hadd(A,v);
		f_vec4(y);
	}
}
BENCHMARK(bm_mv_hadd);

static void bm_mv_transpose_explicit(benchmark::State& state) {
	auto tmp = get_example<4>(); auto A = tmp.first; auto v = tmp.second;

	for (auto _ : state) {
		v.d[0] += 1e-5;
		auto y = mv_transpose_explicit(A,v);
		f_vec4(y);
	}
}
BENCHMARK(bm_mv_transpose_explicit);

static void bm_mv_transpose_implicit(benchmark::State& state) {
	auto tmp = get_example<4>(); auto A = tmp.first; auto v = tmp.second;

	for (auto _ : state) {
		v.d[0] += 1e-5;
		auto y = mv_transpose_implicit(A,v);
		f_vec4(y);
	}
}
BENCHMARK(bm_mv_transpose_implicit);

static void bm_mv_loop(benchmark::State& state) {
	auto tmp = get_example<4>(); auto A = tmp.first; auto v = tmp.second;

	for (auto _ : state) {
		v.d[0] += 1e-5;
		auto y = mv_loop(A,v);
		f_vec4(y);
	}
}
BENCHMARK(bm_mv_loop);

static void bm_mv_eigen(benchmark::State& state) {
	auto tmp = get_example<4>(); auto A = tmp.first; auto v = tmp.second;

	for (auto _ : state) {
		v.d[0] += 1e-5;
		auto y = e::mv_eigen<4>({A.d},{v.d});
		f_ptr(y.d);
	}
}
BENCHMARK(bm_mv_eigen);
*/

#define BM_MV(func) \
template <int N> \
static void bm_ ## func (benchmark::State& state) { \
	auto tmp = get_example<N>(); auto A = tmp.first; auto v = tmp.second; \
 \
	for (auto _ : state) { \
		v[0] += 1e-5; \
		auto y = func (A,v); \
		f_ptr(y.d); \
	} \
} \
BENCHMARK_TEMPLATE(bm_ ## func, 4); \
BENCHMARK_TEMPLATE(bm_ ## func, 8); \
BENCHMARK_TEMPLATE(bm_ ## func, 20);



// Register the function as a benchmark
BENCHMARK(tests)->Iterations(1);

BM_MV(mv_transpose_implicit)
BM_MV(mv_loop)
BM_MV(mv_eigen)

BENCHMARK_MAIN();

