#include <rocsolver/rocsolver.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

// 生成随机实对称矩阵
template <typename T>
void initSymmetricMatrix(T* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            T value = static_cast<T>(rand()) / RAND_MAX; // 随机值
            matrix[i * N + j] = value;
            matrix[j * N + i] = value; // 确保矩阵对称
        }
    }
}

// 打印矩阵
template <typename T>
void printMatrix(const T* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

// 测试 rocsolver_syev 函数
template <typename T>
void testSyev(int N) {
    size_t size = N * N * sizeof(T);

    // 主机端矩阵
    std::vector<T> h_A(N * N);
    std::vector<T> h_eigenvalues(N);
    std::vector<T> h_work(N); // 工作数组
    int h_info = 0; // 状态信息

    // 初始化实对称矩阵
    initSymmetricMatrix(h_A.data(), N);

    // 设备端矩阵
    T *d_A, *d_eigenvalues, *d_work;
    int *d_info;
    hipMalloc(&d_A, size);
    hipMalloc(&d_eigenvalues, N * sizeof(T));
    hipMalloc(&d_work, N * sizeof(T)); // 工作数组
    hipMalloc(&d_info, sizeof(int)); // 状态信息

    // 将数据从主机复制到设备
    hipMemcpy(d_A, h_A.data(), size, hipMemcpyHostToDevice);

    // 创建 rocsolver 句柄
    rocsolver_handle handle;
    rocsolver_create_handle(&handle);

    // 启动计时
    auto start = std::chrono::high_resolution_clock::now();

    // 调用 rocsolver_syev 函数
    if constexpr (std::is_same_v<T, float>) {
        rocsolver_ssyev(handle, rocblas_evect_original, rocblas_fill_upper,
                        N, d_A, N, d_eigenvalues, d_work, d_info);
    } else if constexpr (std::is_same_v<T, double>) {
        rocsolver_dsyev(handle, rocblas_evect_original, rocblas_fill_upper,
                        N, d_A, N, d_eigenvalues, d_work, d_info);
    }

    // 等待 GPU 完成
    hipDeviceSynchronize();

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 将结果从设备复制回主机
    hipMemcpy(h_eigenvalues.data(), d_eigenvalues, N * sizeof(T), hipMemcpyDeviceToHost);
    hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);

    // 检查状态信息
    if (h_info == 0) {
        std::cout << "Matrix size: " << N << "x" << N
                  << ", Time: " << elapsed.count() << " seconds" << std::endl;
        std::cout << "Eigenvalues: ";
        for (int i = 0; i < N; i++) {
            std::cout << h_eigenvalues[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cerr << "Error: rocsolver_syev failed with info = " << h_info << std::endl;
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
    std::vector<int> sizes = {4, 8, 16}; // 矩阵大小

    for (int size : sizes) {
        std::cout << "Testing float matrix (" << size << "x" << size << "):" << std::endl;
        testSyev<float>(size);

        std::cout << "Testing double matrix (" << size << "x" << size << "):" << std::endl;
        testSyev<double>(size);
    }

    return 0;
}

