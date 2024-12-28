# Environment setup
SHELL := /bin/bash

# Directory structure
BUILD_DIR = target
BIN_DIR = $(BUILD_DIR)/bin
RESULTS_DIR = $(BUILD_DIR)/results
REPORTS_DIR = $(BUILD_DIR)/reports

# ROCM settings
ROCM_PATH = /opt/rocm
HIPCC = hipcc
ROCM_CXXFLAGS = -O3 -std=c++14
ROCM_LDFLAGS = -lrocblas
ROCM_SRCS = $(wildcard rocm/*.cpp)
ROCM_BINS = $(ROCM_SRCS:rocm/%.cpp=$(BIN_DIR)/rocm/%)
ROCM_RESULTS = $(ROCM_SRCS:rocm/%.cpp=$(RESULTS_DIR)/rocm_%.txt)

# CUDA settings
CUDA_PATH = /usr/local/cuda
NVCC = nvcc
CUDA_FLAGS = -O3 -std=c++14 -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcublas -lcudart
CUDA_SRCS = $(wildcard cuda/*.cpp)
CUDA_BINS = $(CUDA_SRCS:cuda/%.cpp=$(BIN_DIR)/cuda/%)
CUDA_RESULTS = $(CUDA_SRCS:cuda/%.cpp=$(RESULTS_DIR)/cuda_%.txt)

# MKL settings
MKL_PATH = /opt/intel/oneapi/mkl/latest
ICPX = icpx
MKL_FLAGS = -O3 -std=c++14 -qopenmp -I$(MKL_PATH)/include -L$(MKL_PATH)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
MKL_SRCS = $(wildcard mkl/*.cpp)
MKL_BINS = $(MKL_SRCS:mkl/%.cpp=$(BIN_DIR)/mkl/%)
MKL_RESULTS = $(MKL_SRCS:mkl/%.cpp=$(RESULTS_DIR)/mkl_%.txt)

# BLAS settings
CXX = g++
BLAS_FLAGS = -O3 -std=c++14 -fopenmp -lopenblas
BLAS_SRCS = $(wildcard blas/*.cpp)
BLAS_BINS = $(BLAS_SRCS:blas/%.cpp=$(BIN_DIR)/blas/%)
BLAS_RESULTS = $(BLAS_SRCS:blas/%.cpp=$(RESULTS_DIR)/blas_%.txt)

# All targets
.PHONY: all clean setup rocm cuda mkl blas report

all: setup rocm cuda mkl blas report

# Directory setup
setup:
	@mkdir -p $(BIN_DIR)/rocm $(BIN_DIR)/cuda $(BIN_DIR)/mkl $(BIN_DIR)/blas
	@mkdir -p $(RESULTS_DIR) $(REPORTS_DIR)

# Clean
clean:
	@rm -rf $(BUILD_DIR)

# ROCM rules
rocm: $(ROCM_RESULTS)

$(BIN_DIR)/rocm/%: rocm/%.cpp
	@mkdir -p $(dir $@)
	$(HIPCC) $(ROCM_CXXFLAGS) $< -o $@ $(ROCM_LDFLAGS)

$(RESULTS_DIR)/rocm_%.txt: $(BIN_DIR)/rocm/%
	@mkdir -p $(dir $@)
	./$< > $@

# CUDA rules
cuda: $(CUDA_RESULTS)

$(BIN_DIR)/cuda/%: cuda/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(CUDA_FLAGS) $< -o $@

$(RESULTS_DIR)/cuda_%.txt: $(BIN_DIR)/cuda/%
	@mkdir -p $(dir $@)
	./$< > $@

# MKL rules
mkl: $(MKL_RESULTS)

$(BIN_DIR)/mkl/%: mkl/%.cpp
	@mkdir -p $(dir $@)
	source /opt/intel/oneapi/setvars.sh && $(ICPX) $(MKL_FLAGS) $< -o $@

$(RESULTS_DIR)/mkl_%.txt: $(BIN_DIR)/mkl/%
	@mkdir -p $(dir $@)
	source /opt/intel/oneapi/setvars.sh && ./$< > $@

# BLAS rules
blas: $(BLAS_RESULTS)

$(BIN_DIR)/blas/%: blas/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(BLAS_FLAGS) $< -o $@

$(RESULTS_DIR)/blas_%.txt: $(BIN_DIR)/blas/%
	@mkdir -p $(dir $@)
	./$< > $@

# Generate combined report
report:
	@echo "Generating combined report..."
	@echo "<!DOCTYPE html>" > $(REPORTS_DIR)/combined_report.html
	@echo "<html><head><title>Combined Benchmark Results</title>" >> $(REPORTS_DIR)/combined_report.html
	@echo "<style>pre { background-color: #f5f5f5; padding: 10px; }</style>" >> $(REPORTS_DIR)/combined_report.html
	@echo "</head><body>" >> $(REPORTS_DIR)/combined_report.html
	@echo "<h1>Combined Benchmark Results</h1>" >> $(REPORTS_DIR)/combined_report.html
	@for result in $(RESULTS_DIR)/*.txt; do \
		if [ -f "$$result" ]; then \
			echo "<h2>$$(basename $$result .txt)</h2>" >> $(REPORTS_DIR)/combined_report.html; \
			echo "<pre>" >> $(REPORTS_DIR)/combined_report.html; \
			cat $$result >> $(REPORTS_DIR)/combined_report.html; \
			echo "</pre>" >> $(REPORTS_DIR)/combined_report.html; \
		fi \
	done
	@echo "</body></html>" >> $(REPORTS_DIR)/combined_report.html
