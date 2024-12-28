#include <rocblas/rocblas.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>

// Initialize matrix with random values
void initMatrix(double* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX; // Random values between 0 and 1
    }
}

// Matrix multiplication benchmark
void benchmarkMatrixMul(int N) {
    size_t size = N * N * sizeof(double);

    // Use unified memory
    double *d_A, *d_B, *d_C;
    hipMallocManaged(&d_A, size);
    hipMallocManaged(&d_B, size);
    hipMallocManaged(&d_C, size);

    // Initialize matrices directly on GPU
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < N * N; i++) {
        d_A[i] = static_cast<double>(rand()) / RAND_MAX;
        d_B[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // Create rocBLAS handle
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Set pointer mode to host
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    // Create GPU stream
    hipStream_t stream;
    hipStreamCreate(&stream);
    rocblas_set_stream(handle, stream);

    // Define matrix parameters
    const double alpha = 1.0;
    const double beta = 0.0;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Call rocBLAS matrix multiplication function (FP64)
    rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                  N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    // Synchronize and stop timing
    hipStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Calculate performance metrics
    double seconds = diff.count();
    double flops = 2.0 * N * N * N; // Number of floating point operations
    double gflops = (flops * 1e-9) / seconds; // Convert to GFLOPS

    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Time: " << seconds << " seconds" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    // Cleanup
    rocblas_destroy_handle(handle);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipStreamDestroy(stream);
}

int main() {
    // Test different matrix sizes (2^10 to 2^15)
    std::vector<int> sizes = {8192, 16384, 32768}; // Larger matrices for MI300X

    for (int size : sizes) {
        std::cout << "Testing matrix size: " << size << "x" << size << std::endl;
        benchmarkMatrixMul(size);
        std::cout << "------------------------" << std::endl;
    }
    return 0;
}
