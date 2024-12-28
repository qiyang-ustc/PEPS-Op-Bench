#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Vector size
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Host vectors
    std::vector<float> h_A(N, 1.0f); // Initialize to 1.0
    std::vector<float> h_B(N, 2.0f); // Initialize to 2.0
    std::vector<float> h_C(N);

    // Device vectors
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) {
            success = false;
            break;
        }
    }

    std::cout << "Vector size: " << N << std::endl;
    std::cout << "Time: " << diff.count() << " seconds" << std::endl;
    std::cout << "Test " << (success ? "PASSED" : "FAILED") << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
