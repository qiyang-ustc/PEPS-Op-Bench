#include <rocblas/rocblas.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>

// 初始化矩阵
void initMatrix(double* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = static_cast<double>(rand()) / RAND_MAX; // 随机值
    }
}

// 矩阵乘法基准测试
void benchmarkMatrixMul(int N) {
    size_t size = N * N * sizeof(double);

    // 主机端矩阵
    double *h_A = new double[N * N];
    double *h_B = new double[N * N];
    double *h_C = new double[N * N];

    // 初始化矩阵
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // 设备端矩阵
    double *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);

    // 将数据从主机复制到设备
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    // 创建 rocBLAS 句柄
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // 定义矩阵参数
    const double alpha = 1.0;
    const double beta = 0.0;

    // 启动计时
    auto start = std::chrono::high_resolution_clock::now();

    // 调用 rocBLAS 矩阵乘法函数 (FP64)
    rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                  N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    // 等待 GPU 完成
    hipDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 将结果从设备复制回主机
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);

    // 输出执行时间
    std::cout << "Matrix size: " << N << "x" << N
              << ", Time: " << elapsed.count() << " seconds" << std::endl;

    // 释放资源
    rocblas_destroy_handle(handle);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main() {
    // 测试不同大小的矩阵乘法
    std::vector<int> sizes = {1024, 2048, 4096, 8192}; // 矩阵大小

    for (int size : sizes) {
        benchmarkMatrixMul(size);
    }

    return 0;
}

