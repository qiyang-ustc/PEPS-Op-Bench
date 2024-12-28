#include <rocblas/rocblas.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>

// Initialize matrix with random values
void initMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX; // Random values between 0 and 1
    }
}

// Matrix multiplication benchmark
void benchmarkMatrixMul(int N) {
    size_t size = N * N * sizeof(float);

    // Host matrices
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    // Initialize matrices
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Device matrices
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);

    // Copy data from host to device
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    // Create rocBLAS handle
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Define matrix parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Call rocBLAS matrix multiplication function
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                  N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    // Synchronize and stop timing
    hipDeviceSynchronize();
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
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main() {
    // Test different matrix sizes (2^10 to 2^15)
    std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384, 32768}; // 2^10 to 2^15

    for (int size : sizes) {
        std::cout << "Testing matrix size: " << size << "x" << size << std::endl;
        benchmarkMatrixMul(size);
        std::cout << "------------------------" << std::endl;
    }
    return 0;
}
