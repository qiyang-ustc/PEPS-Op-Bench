#!/bin/bash

# Compiler settings
CXX=g++
CXXFLAGS="-O3 -std=c++14 -fopenmp"
LDFLAGS="-lopenblas"

# Build all tests
build_tests() {
    $CXX $CXXFLAGS matrix_mul_benchmark.cpp -o matrix_mul_test $LDFLAGS
    $CXX $CXXFLAGS vector_add.cpp -o vector_add_test
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
    <title>CPU Test Results</title>
</head>
<body>
    <h1>CPU Test Results</h1>
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