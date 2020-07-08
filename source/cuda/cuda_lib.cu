#include <iostream>
#include <cuda.h>
#include <math.h>

#include "./cuda_lib.hpp"

#define ADD 0
#define SUB 1
#define MUL 2
#define DIV 3

#define round_up 0
#define round_down 1
#define round_to_nearest 2
#define round_toward_zero 3

const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 16;

__global__
void matrixMultiplicationKernel(const int* d_matrixA,
                                const int* d_matrixB,
                                int        N,
                                int*       d_matrixC) {
    int Row = blockIdx.y*blockDim.y + threadIdx.y;
    int Col = blockIdx.x*blockDim.x + threadIdx.x;
    
    int Pvalue = 0;
    if (Row < N && Col < N) {
        for (int k = 0; k < N; ++k)
            Pvalue += d_matrixA[Row*N+k] * d_matrixB[Col+k*N];        

        d_matrixC[Row*N+Col] = Pvalue;
    }
}

__global__
void cuda_operation_float(float first_value, float second_value, int operation, float * res) {
    switch (operation) {
        case ADD:
            * res = first_value + second_value;
            break;
        case SUB:
            * res = first_value - second_value;
            break;
        case MUL:
            * res = first_value * second_value;
            break;
        case DIV:
            * res = first_value / second_value;
            break;
    }
}

__global__
void cuda_operation_double(double first_value, double second_value, int operation, double * res) {
    switch (operation) {
        case ADD:
            * res = first_value + second_value;
            break;
        case SUB:
            * res = first_value - second_value;
            break;
        case MUL:
            * res = first_value * second_value;
            break;
        case DIV:
            * res = first_value / second_value;
            break;
    }
}

void ariadne_cuda::function(const int N, int * h_matrixA, int * h_matrixB, int * h_matrixC) {
    int *d_matrixA, *d_matrixB, *d_matrixC;
    cudaMalloc( &d_matrixA, N*N * sizeof(int) );
    cudaMalloc( &d_matrixB, N*N * sizeof(int) );
    cudaMalloc( &d_matrixC, N*N * sizeof(int) );

    cudaMemcpy( d_matrixA, h_matrixA, N*N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_matrixB, h_matrixB, N*N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 DimGrid(N/BLOCK_SIZE_X, N/BLOCK_SIZE_Y, 1);
    if (N%BLOCK_SIZE_X) DimGrid.x++;
    if (N%BLOCK_SIZE_Y) DimGrid.y++;
    dim3 DimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    
    matrixMultiplicationKernel<<< DimGrid,DimBlock>>> (d_matrixA, d_matrixB, N, d_matrixC);

    cudaMemcpy( h_matrixC, d_matrixC, N*N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "GPU: " << std::endl;
    for (int i = 0; i < N * N; i++){
        if (i % N == 0){
            std::cout << std::endl;
        }
        std::cout << h_matrixC[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(&d_matrixA);
    cudaFree(&d_matrixB);
    cudaFree(&d_matrixC);
}

float ariadne_cuda::float_approximation (float first_value, float second_value, int operation, int rounding) {
    float * res_d;
    float * res_h = new float();

    cudaMalloc(&res_d, sizeof(float));
    cuda_operation_float <<< 1, 1 >>> (first_value, second_value, operation, res_d);

    cudaMemcpy(res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(res_d);

    switch (operation) {
        case round_down:
            
            break;
        case round_up:
            
            break;
        case round_to_nearest:
            
            break;
        case round_toward_zero:
            
            break;
    }
    
    return * res_h;
}

double ariadne_cuda::double_approximation (double first_value, double second_value, int operation, int rounding) {
    double * res_d;
    double * res_h = new double();

    cudaMalloc(&res_d, sizeof(double));
    cuda_operation_double <<< 1, 1 >>> (first_value, second_value, operation, res_d);

    cudaMemcpy(res_h, res_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(res_d);

    switch (operation) {
        case round_down:
            
            break;
        case round_up:
            
            break;
        case round_to_nearest:
            
            break;
        case round_toward_zero:
            
            break;
    }
    
    return * res_h;
}
