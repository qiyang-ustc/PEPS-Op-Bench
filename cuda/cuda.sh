#!/bin/bash

# Compiler settings
NVCC=nvcc
CXXFLAGS="-O3 -std=c++14"
LDFLAGS="-lcublas"

# Build all tests
build_tests() {
    $NVCC $CXXFLAGS matrix_mul_benchmark.cpp -o matrix_mul_test $LDFLAGS
    $NVCC $CXXFLAGS vector_add.cpp -o vector_add_test
}

# Run tests and generate reports
run_tests() {
    echo "Running Matrix Multiplication Benchmark..."
    ./matrix_mul_test > results_matrix.txt
    
    echo "Running Vector Addition Test..."
    ./vector_add_test > results_vector_add.txt
}

# Generate HTML report
generate_report() {
    echo "Generating test report..."
    cat > report.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>CUDA Test Results</title>
</head>
<body>
    <h1>CUDA Test Results</h1>
    <h2>Matrix Multiplication</h2>
    <pre>\$(cat results_matrix.txt)</pre>
    
    <h2>Vector Addition</h2>
    <pre>\$(cat results_vector_add.txt)</pre>
</body>
</html>
EOF
}

# Main execution
build_tests
run_tests
generate_report