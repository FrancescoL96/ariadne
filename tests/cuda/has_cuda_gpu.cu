#include <iostream>
#include <cuda_runtime.h>

#include "../../source/cuda/CheckError.cuh"

int main() {
    int device_count, device;
    int gpu_device_count = 0;
    struct cudaDeviceProp properties;
    SAFE_CALL(cudaGetDeviceCount(&device_count));
    
    for (device = 0; device < device_count; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999)
            gpu_device_count++;
    }
    printf("%d GPU CUDA device(s) found", gpu_device_count);

    if (gpu_device_count > 0)
        return 0;
    else
        return 1;
}