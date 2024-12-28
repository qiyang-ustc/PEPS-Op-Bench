# PEPS-Op-Bench

A comprehensive benchmark suite for comparing matrix multiplication (GEMM) performance across different hardware platforms and frameworks.

## Hardware Theoretical Peak Performance (FP64)

| Hardware    | Peak Performance |
|------------|------------------|
| NVIDIA A100 | 19.5 TFLOPS     |
| NVIDIA H100 | 67 TFLOPS       |
| AMD MI300A  | 122.6 TFLOPS    |

## Benchmark Results

### NVIDIA CUDA - A100 GPU (FP64)

| Matrix Size | Time (ms) | Performance (GFLOPS) |
|------------|-----------|---------------------|
| 1024x1024  | 0.151     | 14,234.3           |
| 2048x2048  | 0.962     | 17,866.9           |
| 4096x4096  | 7.139     | 19,251.6           |
| 8192x8192  | 56.984    | 19,295.0           |

### NVIDIA CUDA - H100 GPU (FP64)

| Matrix Size | Time (ms) | Performance (GFLOPS) |
|------------|-----------|---------------------|
| 1024x1024  | 0.061     | 35,437.0           |
| 2048x2048  | 0.301     | 57,108.6           |
| 4096x4096  | 2.244     | 61,260.4           |
| 8192x8192  | 30.863    | 35,625.5           |

### AMD MI300A (FP64)

| Matrix Size   | Time (s)  | Performance (GFLOPS) |
|--------------|-----------|---------------------|
| 1024x1024    | 0.024     | 89.3               |
| 2048x2048    | 0.000461  | 37,228.0           |
| 4096x4096    | 0.001770  | 77,668.5           |
| 8192x8192    | 0.018798  | 58,490.4           |
| 16384x16384  | 0.140926  | 62,416.6           |
| 32768x32768  | 1.920930  | 36,632.7           |

### Intel MKL - 16 Cores CPU (FP64)

| Matrix Size   | Time (s)  | Performance (GFLOPS) |
|--------------|-----------|---------------------|
| 1024x1024    | 0.030     | 71.4               |
| 2048x2048    | 0.023     | 731.3              |
| 4096x4096    | 0.188     | 729.3              |
| 8192x8192    | 1.486     | 739.9              |
| 16384x16384  | 11.915    | 738.3              |
| 32768x32768  | 103.315   | 681.1              |

## Key Observations

1. The NVIDIA H100 shows significant performance improvements over the A100, with peak performance reaching ~61 TFLOPS (91% of theoretical peak).
2. AMD MI300A achieves peak performance of ~77 TFLOPS, which is about 63% of its theoretical peak of 122.6 TFLOPS, indicating room for optimization.
3. Intel MKL on CPU provides consistent performance around 730-740 GFLOPS for mid-range matrix sizes.
4. All platforms show some performance degradation at larger matrix sizes (32768x32768).