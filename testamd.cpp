#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    std::cout << "Number of HIP devices: " << deviceCount << std::endl;
    return 0;
}
