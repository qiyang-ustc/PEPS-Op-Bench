#include <iostream>
#include <vector>
#include <chrono>
#include <rocblas.h>
#include <rocsolver.h>

#define ROCBLAS_CHECK(status) \
    if (status != rocblas_status_success) { \
        std::cerr << "rocBLAS error: " << status << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define ROCSOLVER_CHECK(status) \
    if (status != rocblas_status_success) { \
        std::cerr << "rocSOLVER error: " << status << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

void benchmark_dgesvd(int m, int n) {
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    rocblas_int lda = m;
    rocblas_int ldu = m;
    rocblas_int ldv = n;

    std::vector<double> A(m * n);
    std::vector<double> S(std::min(m, n));
    std::vector<double> U(m * m);
    std::vector<double> V(n * n);
    std::vector<double> E(std::min(m, n) - 1);
    rocblas_int info;

    // Initialize matrix A with random values
    for (int i = 0; i < m * n; ++i) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // Warm-up
    ROCSOLVER_CHECK(rocsolver_dgesvd(
        handle, rocblas_svect_all, rocblas_svect_all, m, n, A.data(), lda, S.data(), U.data(), ldu, V.data(), ldv, E.data(), rocblas_workmode_auto, &info
    ));

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    ROCSOLVER_CHECK(rocsolver_dgesvd(
        handle, rocblas_svect_all, rocblas_svect_all, m, n, A.data(), lda, S.data(), U.data(), ldu, V.data(), ldv, E.data(), rocblas_workmode_auto, &info
    ));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Matrix size: " << m << "x" << n << ", Time: " << elapsed_seconds.count() << " seconds" << std::endl;

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

int main() {
    std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384};

    for (int size : sizes) {
        benchmark_dgesvd(size, size);
    }

    return 0;
}

