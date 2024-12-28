# Environment setup
SHELL := /bin/bash

# ROCM settings
ROCM_PATH = /opt/rocm
HIPCC = hipcc
ROCM_FLAGS = -O3 -std=c++14 -I$(ROCM_PATH)/include -L$(ROCM_PATH)/lib -lrocblas -lhip_hcc

# CUDA settings
CUDA_PATH = /usr/local/cuda
NVCC = nvcc
CUDA_FLAGS = -O3 -std=c++14 -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcublas -lcudart

# MKL settings
MKL_PATH = /opt/intel/oneapi/mkl/latest
ICPX = icpx
MKL_FLAGS = -O3 -std=c++14 -qopenmp -I$(MKL_PATH)/include -L$(MKL_PATH)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

# BLAS settings
CXX = g++
BLAS_FLAGS = -O3 -std=c++14 -fopenmp -lopenblas

# Output directories
RESULTS_DIR = target
REPORTS_DIR = target

# Targets
all: setup rocm cuda mkl blas report

setup:
	mkdir -p $(RESULTS_DIR) $(REPORTS_DIR)

# ROCM targets
rocm: rocm/matrix_mul_test rocm/matrix_mul_fp64_test rocm/vector_add_test rocm/basic_test
	@echo "Running ROCM tests..."
	./rocm/matrix_mul_test > $(RESULTS_DIR)/rocm_matrix_fp32.txt
	./rocm/matrix_mul_fp64_test > $(RESULTS_DIR)/rocm_matrix_fp64.txt
	./rocm/vector_add_test > $(RESULTS_DIR)/rocm_vector_add.txt
	./rocm/basic_test > $(RESULTS_DIR)/rocm_basic.txt

rocm/matrix_mul_test: rocm/matrix_mul_benchmark.cpp
	$(HIPCC) $(ROCM_FLAGS) $< -o $@

rocm/matrix_mul_fp64_test: rocm/matrix_mul_fp64_benchmark.cpp
	$(HIPCC) $(ROCM_FLAGS) $< -o $@

rocm/vector_add_test: rocm/vector_add.cpp
	$(HIPCC) $(ROCM_FLAGS) $< -o $@

rocm/basic_test: rocm/testamd.cpp
	$(HIPCC) $(ROCM_FLAGS) $< -o $@

# CUDA targets
cuda: cuda/matrix_mul_test cuda/vector_add_test
	@echo "Running CUDA tests..."
	./cuda/matrix_mul_test > $(RESULTS_DIR)/cuda_matrix.txt
	./cuda/vector_add_test > $(RESULTS_DIR)/cuda_vector_add.txt

cuda/matrix_mul_test: cuda/matrix_mul_benchmark.cpp
	$(NVCC) $(CUDA_FLAGS) $< -o $@

cuda/vector_add_test: cuda/vector_add.cpp
	$(NVCC) $(CUDA_FLAGS) $< -o $@

# MKL targets
mkl: mkl/matrix_mul_test mkl/vector_add_test
	@echo "Running MKL tests..."
	source /opt/intel/oneapi/setvars.sh && \
	./mkl/matrix_mul_test > $(RESULTS_DIR)/mkl_matrix.txt && \
	./mkl/vector_add_test > $(RESULTS_DIR)/mkl_vector_add.txt

mkl/matrix_mul_test: mkl/matrix_mul_benchmark.cpp
	source /opt/intel/oneapi/setvars.sh && \
	$(ICPX) $(MKL_FLAGS) $< -o $@

mkl/vector_add_test: mkl/vector_add.cpp
	source /opt/intel/oneapi/setvars.sh && \
	$(ICPX) $(MKL_FLAGS) $< -o $@

# BLAS targets
blas: blas/matrix_mul_test blas/vector_add_test
	@echo "Running BLAS tests..."
	./blas/matrix_mul_test > $(RESULTS_DIR)/blas_matrix.txt
	./blas/vector_add_test > $(RESULTS_DIR)/blas_vector_add.txt

blas/matrix_mul_test: blas/matrix_mul_benchmark.cpp
	$(CXX) $(BLAS_FLAGS) $< -o $@

blas/vector_add_test: blas/vector_add.cpp
	$(CXX) $(BLAS_FLAGS) $< -o $@

# Generate combined report
report:
	@echo "Generating combined report..."
	@echo "<!DOCTYPE html>" > $(REPORTS_DIR)/combined_report.html
	@echo "<html><head><title>Combined Benchmark Results</title>" >> $(REPORTS_DIR)/combined_report.html
	@echo "<style>pre { background-color: #f5f5f5; padding: 10px; }</style>" >> $(REPORTS_DIR)/combined_report.html
	@echo "</head><body>" >> $(REPORTS_DIR)/combined_report.html
	@echo "<h1>Combined Benchmark Results</h1>" >> $(REPORTS_DIR)/combined_report.html
	@echo "<h2>ROCM Results</h2>" >> $(REPORTS_DIR)/combined_report.html
	@echo "<h3>Matrix Multiplication FP32</h3><pre>" >> $(REPORTS_DIR)/combined_report.html
	@cat $(RESULTS_DIR)/rocm_matrix_fp32.txt >> $(REPORTS_DIR)/combined_report.html
	@echo "</pre><h3>Matrix Multiplication FP64</h3><pre>" >> $(REPORTS_DIR)/combined_report.html
	@cat $(RESULTS_DIR)/rocm_matrix_fp64.txt >> $(REPORTS_DIR)/combined_report.html
	@echo "</pre><h3>Vector Addition</h3><pre>" >> $(REPORTS_DIR)/combined_report.html
	@cat $(RESULTS_DIR)/rocm_vector_add.txt >> $(REPORTS_DIR)/combined_report.html
	@echo "</pre><h2>CUDA Results</h2>" >> $(REPORTS_DIR)/combined_report.html
	@echo "<h3>Matrix Multiplication</h3><pre>" >> $(REPORTS_DIR)/combined_report.html
	@cat $(RESULTS_DIR)/cuda_matrix.txt >> $(REPORTS_DIR)/combined_report.html
	@echo "</pre><h3>Vector Addition</h3><pre>" >> $(REPORTS_DIR)/combined_report.html
	@cat $(RESULTS_DIR)/cuda_vector_add.txt >> $(REPORTS_DIR)/combined_report.html
	@echo "</pre><h2>MKL Results</h2>" >> $(REPORTS_DIR)/combined_report.html
	@echo "<h3>Matrix Multiplication</h3><pre>" >> $(REPORTS_DIR)/combined_report.html
	@cat $(RESULTS_DIR)/mkl_matrix.txt >> $(REPORTS_DIR)/combined_report.html
	@echo "</pre><h3>Vector Addition</h3><pre>" >> $(REPORTS_DIR)/combined_report.html
	@cat $(RESULTS_DIR)/mkl_vector_add.txt >> $(REPORTS_DIR)/combined_report.html
	@echo "</pre><h2>BLAS Results</h2>" >> $(REPORTS_DIR)/combined_report.html
	@echo "<h3>Matrix Multiplication</h3><pre>" >> $(REPORTS_DIR)/combined_report.html
	@cat $(RESULTS_DIR)/blas_matrix.txt >> $(REPORTS_DIR)/combined_report.html
	@echo "</pre><h3>Vector Addition</h3><pre>" >> $(REPORTS_DIR)/combined_report.html
	@cat $(RESULTS_DIR)/blas_vector_add.txt >> $(REPORTS_DIR)/combined_report.html
	@echo "</pre></body></html>" >> $(REPORTS_DIR)/combined_report.html

clean:
	rm -f rocm/matrix_mul_test rocm/matrix_mul_fp64_test rocm/vector_add_test rocm/basic_test
	rm -f cuda/matrix_mul_test cuda/vector_add_test
	rm -f mkl/matrix_mul_test mkl/vector_add_test
	rm -f blas/matrix_mul_test blas/vector_add_test
	rm -rf $(RESULTS_DIR) $(REPORTS_DIR)

.PHONY: all setup rocm cuda mkl blas report clean
