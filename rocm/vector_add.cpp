#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// HIP 核函数：向量加法
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // 向量大小
    const int N = 1 << 20; // 1M 元素
    const size_t size = N * sizeof(float);

    // 主机端向量
    std::vector<float> h_A(N, 1.0f); // 初始化为 1.0
    std::vector<float> h_B(N, 2.0f); // 初始化为 2.0
    std::vector<float> h_C(N);

    // 设备端向量
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);

    // 将数据从主机复制到设备
    hipMemcpy(d_A, h_A.data(), size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), size, hipMemcpyHostToDevice);

    // 定义线程块和网格大小
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // 启动核函数
    hipLaunchKernelGGL(vectorAdd, dim3(gridSize), dim3(blockSize), 0, 0, d_A, d_B, d_C, N);

    // 将结果从设备复制回主机
    hipMemcpy(h_C.data(), d_C, size, hipMemcpyDeviceToHost);

    // 验证结果
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Vector addition successful! All values are correct." << std::endl;
    } else {
        std::cout << "Vector addition failed! Some values are incorrect." << std::endl;
    }

    // 释放设备内存
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;
}

