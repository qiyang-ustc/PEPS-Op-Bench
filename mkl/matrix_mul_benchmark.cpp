#include <mkl.h>
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
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    // Initialize matrices
    initMatrix(A, N);
    initMatrix(B, N);

    // Define matrix parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const MKL_INT lda = N;
    const MKL_INT ldb = N;
    const MKL_INT ldc = N;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Call MKL matrix multiplication function
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, alpha, A, lda, B, ldb, beta, C, ldc);

    // Stop timing
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
    delete[] A;
    delete[] B;
    delete[] C;
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
