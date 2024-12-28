#!/bin/bash
# build_and_test.sh

# Compiler settings
HIPCC=hipcc
CXXFLAGS="-O3 -std=c++14"
LDFLAGS="-lrocblas"

# Build all tests
build_tests() {
    $HIPCC $CXXFLAGS matrix_mul_benchmark.cpp -o matrix_mul_test $LDFLAGS
    $HIPCC $CXXFLAGS matrix_mul_fp64_benchmark.cpp -o matrix_mul_fp64_test $LDFLAGS
    $HIPCC $CXXFLAGS vector_add.cpp -o vector_add_test
    $HIPCC $CXXFLAGS testamd.cpp -o basic_test
}

# Run tests and generate reports
run_tests() {
    echo "Running Matrix Multiplication FP32 Benchmark..."
    ./matrix_mul_test > results_matrix_fp32.txt
    
    echo "Running Matrix Multiplication FP64 Benchmark..."
    ./matrix_mul_fp64_test > results_matrix_fp64.txt
    
    echo "Running Vector Addition Test..."
    ./vector_add_test > results_vector_add.txt
    
    echo "Running Basic AMD GPU Test..."
    ./basic_test > results_basic.txt
}

# Generate HTML report
generate_report() {
    echo "Generating test report..."
    cat > report.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>ROCm Test Results</title>
</head>
<body>
    <h1>ROCm Test Results</h1>
    <h2>Matrix Multiplication FP32</h2>
    <pre>$(cat results_matrix_fp32.txt)</pre>
    
    <h2>Matrix Multiplication FP64</h2>
    <pre>$(cat results_matrix_fp64.txt)</pre>
    
    <h2>Vector Addition</h2>
    <pre>$(cat results_vector_add.txt)</pre>
    
    <h2>Basic AMD GPU Test</h2>
    <pre>$(cat results_basic.txt)</pre>
</body>
</html>
EOF
}

# Main execution
build_tests
run_tests
generate_report