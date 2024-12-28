#include <rocsolver/rocsolver.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

// 生成随机实对称矩阵
void initSymmetricMatrix(double* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            double value = static_cast<double>(rand()) / RAND_MAX; // 随机值
            matrix[i * N + j] = value;
            matrix[j * N + i] = value; // 确保矩阵对称
        }
    }
}

// 基准测试函数
void benchmarkSyevd(int N) {
    size_t size = N * N * sizeof(double);

    // 主机端矩阵
    std::vector<double> h_A(N * N);
    std::vector<double> h_eigenvalues(N);
    std::vector<double> h_work(N); // 工作数组
    int h_info = 0; // 状态信息

    // 初始化实对称矩阵
    initSymmetricMatrix(h_A.data(), N);

    // 设备端矩阵
    double *d_A, *d_eigenvalues, *d_work;
    int *d_info;
    hipMalloc(&d_A, size);
    hipMalloc(&d_eigenvalues, N * sizeof(double));
    hipMalloc(&d_work, N * sizeof(double)); // 工作数组
    hipMalloc(&d_info, sizeof(int)); // 状态信息

    // 将数据从主机复制到设备
    hipMemcpy(d_A, h_A.data(), size, hipMemcpyHostToDevice);

    // 创建 rocsolver 句柄
    rocsolver_handle handle;
    rocsolver_create_handle(&handle);

    // 预热 GPU（可选）
    for (int i = 0; i < 3; i++) {
        rocsolver_dsyevd(handle, rocblas_evect_original, rocblas_fill_upper,
                         N, d_A, N, d_eigenvalues, d_work, d_info);
    }
    hipDeviceSynchronize();

    // 启动计时
    auto start = std::chrono::high_resolution_clock::now();

    // 调用 rocsolver_dsyevd 函数
    rocsolver_dsyevd(handle, rocblas_evect_original, rocblas_fill_upper,
                     N, d_A, N, d_eigenvalues, d_work, d_info);

    // 等待 GPU 完成
    hipDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 将结果从设备复制回主机
    hipMemcpy(h_eigenvalues.data(), d_eigenvalues, N * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);

    // 检查状态信息
    if (h_info == 0) {
        std::cout << "Matrix size: " << N << "x" << N
                  << ", Time: " << elapsed.count() << " seconds" << std::endl;
    } else {
        std::cerr << "Error: rocsolver_dsyevd failed with info = " << h_info << std::endl;
    }

    // 释放资源
    rocsolver_destroy_handle(handle);
    hipFree(d_A);
    hipFree(d_eigenvalues);
    hipFree(d_work);
    hipFree(d_info);
}

int main() {
    // 测试不同大小的矩阵
    std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384}; // 矩阵大小

    for (int size : sizes) {
        std::cout << "Benchmarking matrix size: " << size << "x" << size << std::endl;
        benchmarkSyevd(size);
    }

    return 0;
}

