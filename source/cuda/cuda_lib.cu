#include <stdio.h>
#include <cuda.h>
#include "./include/cuda_lib.hpp"

const int BLOCK_SIZE_X = 1;
const int BLOCK_SIZE_Y = 1;

__global__
void kernel_function(const int num) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row == 0 && col == 0) {
        printf("num: %i\n", num);
    }
}

void function() {
    kernel_function<<< BLOCK_SIZE_X,BLOCK_SIZE_Y>>> (0);
    cudaDeviceReset();
}
