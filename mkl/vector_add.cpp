#include <mkl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

void vectorAdd(const float* A, const float* B, float* C, int N) {
    // Use MKL's BLAS Level 1 routine for vector addition
    // C = A + B is equivalent to C = 1.0 * A + 1.0 * B
    const float alpha = 1.0f;
    const int incx = 1;
    const int incy = 1;
    
    cblas_scopy(N, A, incx, C, incy);  // C = A
    cblas_saxpy(N, alpha, B, incx, C, incy);  // C = alpha * B + C
}

int main() {
    // Vector size
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate vectors with MKL aligned allocator
    float* A = (float*)mkl_malloc(size, 64);
    float* B = (float*)mkl_malloc(size, 64);
    float* C = (float*)mkl_malloc(size, 64);

    if (A == nullptr || B == nullptr || C == nullptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }

    // Initialize vectors
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Print system information
    std::cout << "Number of threads: " << mkl_get_max_threads() << std::endl;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Perform vector addition
    vectorAdd(A, B, C, N);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (std::abs(C[i] - 3.0f) > 1e-6) {
            success = false;
            break;
        }
    }

    std::cout << "Vector size: " << N << std::endl;
    std::cout << "Time: " << diff.count() << " seconds" << std::endl;
    std::cout << "Memory bandwidth: " << (3.0 * size) / (diff.count() * 1e9) << " GB/s" << std::endl;
    std::cout << "Test " << (success ? "PASSED" : "FAILED") << std::endl;

    // Cleanup
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}
