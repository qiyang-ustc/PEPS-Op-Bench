#include <iostream>
#include <vector>
#include <chrono>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

// Function to initialize a matrix with random values
void initialize_matrix(double* A, rocblas_int m, rocblas_int n) {
    for (rocblas_int i = 0; i < m * n; ++i) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

int main() {
    // Initialize rocBLAS and rocSOLVER
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Matrix dimensions
    rocblas_int m = 1000; // Number of rows
    rocblas_int n = 1000; // Number of columns

    // Leading dimension of A
    rocblas_int lda = m;

    // Allocate memory for matrix A
    std::vector<double> A(m * n);
    initialize_matrix(A.data(), m, n);

    // Parameters for singular value decomposition
    rocblas_svect left_svect = rocblas_svect_none; // Do not compute left singular vectors
    rocblas_svect right_svect = rocblas_svect_none; // Do not compute right singular vectors
    rocblas_srange srange = rocblas_srange_all; // Compute all singular values

    // Range for singular values
    double vl = 0.0;
    double vu = 1.0;
    rocblas_int il = 1;
    rocblas_int iu = std::min(m, n);

    // Output arrays
    rocblas_int nsv = 0;
    std::vector<double> S(std::min(m, n));
    std::vector<double> U(m * m); // Allocate memory for U (m x m)
    std::vector<double> V(n * n); // Allocate memory for V (n x n)
    rocblas_int ldu = m; // Leading dimension of U
    rocblas_int ldv = n; // Leading dimension of V
    std::vector<rocblas_int> ifail(std::min(m, n));
    rocblas_int info = 0;

    // Debugging: Print parameters
    std::cout << "Matrix dimensions: m = " << m << ", n = " << n << std::endl;
    std::cout << "Leading dimensions: lda = " << lda << ", ldu = " << ldu << ", ldv = " << ldv << std::endl;
    std::cout << "Singular value range: vl = " << vl << ", vu = " << vu << ", il = " << il << ", iu = " << iu << std::endl;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Call rocsolver_dgesvdx
    rocblas_status status = rocsolver_dgesvdx(
        handle, left_svect, right_svect, srange, m, n, A.data(), lda,
        vl, vu, il, iu, &nsv, S.data(), U.data(), ldu, V.data(), ldv,
        ifail.data(), &info
    );

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Check for errors
    if (status != rocblas_status_success) {
        std::cerr << "Error: rocsolver_dgesvdx failed with status " << status << std::endl;
        std::cerr << "Possible cause: Invalid input parameters." << std::endl;
        return 1;
    }

    // Print the execution time
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    // Clean up
    rocblas_destroy_handle(handle);

    return 0;
}

