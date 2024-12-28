#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

void vectorAdd(const float* A, const float* B, float* C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Vector size
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    // Allocate vectors
    std::vector<float> A(N, 1.0f); // Initialize to 1.0
    std::vector<float> B(N, 2.0f); // Initialize to 2.0
    std::vector<float> C(N);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Perform vector addition
    vectorAdd(A.data(), B.data(), C.data(), N);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (C[i] != 3.0f) {
            success = false;
            break;
        }
    }

    std::cout << "Vector size: " << N << std::endl;
    std::cout << "Time: " << diff.count() << " seconds" << std::endl;
    std::cout << "Test " << (success ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Number of threads: " << omp_get_max_threads() << std::endl;

    return 0;
}
