#include <iostream>
#include <vector>
#include <chrono>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <hip/hip_runtime.h>

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

#define HIP_CHECK(status) \
    if (status != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

void benchmark_dgesvd(int m, int n) {
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    rocblas_int lda = m;
    rocblas_int ldu = m;
    rocblas_int ldv = n;

    // Allocate GPU memory
    double *d_A, *d_S, *d_U, *d_V, *d_E;
    HIP_CHECK(hipMalloc(&d_A, m * n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_S, std::min(m, n) * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_U, m * m * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_V, n * n * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_E, (std::min(m, n) - 1) * sizeof(double)));

    // Initialize matrix A with random values
    std::vector<double> A(m * n);
    for (int i = 0; i < m * n; ++i) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    HIP_CHECK(hipMemcpy(d_A, A.data(), m * n * sizeof(double), hipMemcpyHostToDevice));

    rocblas_int info;

    // Warm-up
    ROCSOLVER_CHECK(rocsolver_dgesvd(
        handle,                  // rocblas_handle
        rocblas_svect_all,       // left_svect
        rocblas_svect_all,       // right_svect
        m,                       // m
        n,                       // n
        d_A,                     // A
        lda,                     // lda
        d_S,                     // S
        d_U,                     // U
        ldu,                     // ldu
        d_V,                     // V
        ldv,                     // ldv
        d_E,                     // E
        rocblas_outofplace,      // fast_alg (use out-of-place computation)
        &info                    // info
    ));

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    ROCSOLVER_CHECK(rocsolver_dgesvd(
        handle,                  // rocblas_handle
        rocblas_svect_all,       // left_svect
        rocblas_svect_all,       // right_svect
        m,                       // m
        n,                       // n
        d_A,                     // A
        lda,                     // lda
        d_S,                     // S
        d_U,                     // U
        ldu,                     // ldu
        d_V,                     // V
        ldv,                     // ldv
        d_E,                     // E
        rocblas_outofplace,      // fast_alg (use out-of-place computation)
        &info                    // info
    ));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Matrix size: " << m << "x" << n << ", Time: " << elapsed_seconds.count() << " seconds" << std::endl;

    // Free GPU memory
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_S));
    HIP_CHECK(hipFree(d_U));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_E));

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

int main() {
    std::vector<int> sizes = {1024}; // Start with smaller sizes

    for (int size : sizes) {
        benchmark_dgesvd(size, size);
    }

    return 0;
}

