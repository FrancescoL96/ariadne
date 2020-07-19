#include <iostream>
#include <cuda.h>
#include <math.h>

#include "./cuda_lib.hpp"
#include "CheckError.cuh"

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
void cuda_operation_float_ru (float first_value, float second_value, int operation, float * res) {
    switch (operation) {
        case ADD:
            * res = __fadd_ru(first_value, second_value);
            break;
        case SUB:
            * res = __fadd_ru(first_value, -second_value);
            break;
        case MUL:
            * res = __fmul_ru (first_value, second_value);
            break;
        case DIV:
            * res = __fdiv_ru (first_value, second_value);
            break;
    }
}

__global__
void cuda_operation_float_rd (float first_value, float second_value, int operation, float * res) {
    switch (operation) {
        case ADD:
            * res = __fadd_rd(first_value, second_value);
            break;
        case SUB:
            * res = __fadd_rd(first_value, -second_value);
            break;
        case MUL:
            * res = __fmul_rd(first_value, second_value);
            break;
        case DIV:
            * res = __fdiv_rd(first_value, second_value);
            break;
    }
}

__global__
void cuda_operation_double_ru(double first_value, double second_value, int operation, double * res) {
    switch (operation) {
        case ADD:
            * res = __dadd_ru(first_value, second_value);
            break;
        case SUB:
            * res = __dadd_ru(first_value, -second_value);
            break;
        case MUL:
            * res = __dmul_ru(first_value, second_value);
            break;
        case DIV:
            * res = __ddiv_ru(first_value, second_value);
            break;
    }
}

__global__
void cuda_operation_double_rd(double first_value, double second_value, int operation, double * res) {
    switch (operation) {
        case ADD:
            * res = __dadd_rd(first_value, second_value);
            break;
        case SUB:
            * res = __dadd_rd(first_value, -second_value);
            break;
        case MUL:
            * res = __dmul_rd(first_value, second_value);
            break;
        case DIV:
            * res = __ddiv_rd(first_value, second_value);
            break;
    }
}

void ariadne_cuda::function(const int N, int * h_matrixA, int * h_matrixB, int * h_matrixC) {
    int *d_matrixA, *d_matrixB, *d_matrixC;
    SAFE_CALL(cudaMalloc( &d_matrixA, N*N * sizeof(int) ));
    SAFE_CALL(cudaMalloc( &d_matrixB, N*N * sizeof(int) ));
    SAFE_CALL(cudaMalloc( &d_matrixC, N*N * sizeof(int) ));

    SAFE_CALL(cudaMemcpy( d_matrixA, h_matrixA, N*N * sizeof(int), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy( d_matrixB, h_matrixB, N*N * sizeof(int), cudaMemcpyHostToDevice));

    dim3 DimGrid(N/BLOCK_SIZE_X, N/BLOCK_SIZE_Y, 1);
    if (N%BLOCK_SIZE_X) DimGrid.x++;
    if (N%BLOCK_SIZE_Y) DimGrid.y++;
    dim3 DimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
    
    matrixMultiplicationKernel<<< DimGrid,DimBlock>>> (d_matrixA, d_matrixB, N, d_matrixC);
    CHECK_CUDA_ERROR

    SAFE_CALL(cudaMemcpy( h_matrixC, d_matrixC, N*N * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "GPU: " << std::endl;
    for (int i = 0; i < N * N; i++){
        if (i % N == 0){
            std::cout << std::endl;
        }
        std::cout << h_matrixC[i] << " ";
    }
    std::cout << std::endl;

    SAFE_CALL(cudaFree(d_matrixA));
    SAFE_CALL(cudaFree(d_matrixB));
    SAFE_CALL(cudaFree(d_matrixC));
}

float ariadne_cuda::float_approximation (float first_value, float second_value, int operation, int rounding) {
    float * res_d;
    float * res_h = new float();

    SAFE_CALL(cudaMalloc(&res_d, sizeof(float)));
    switch (rounding) {
        case round_down:
            cuda_operation_float_rd <<< 1, 1 >>> (first_value, second_value, operation, res_d);
            break;
        case round_up:
            cuda_operation_float_ru <<< 1, 1 >>> (first_value, second_value, operation, res_d);
            break;
        case round_to_nearest:
            
            break;
        case round_toward_zero:
            
            break;
    }
    CHECK_CUDA_ERROR

    SAFE_CALL(cudaMemcpy(res_h, res_d, sizeof(float), cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaFree(res_d));
    
    return * res_h;
}

double ariadne_cuda::double_approximation (double first_value, double second_value, int operation, int rounding) {
    double * res_d;
    double * res_h = new double();

    SAFE_CALL(cudaMalloc(&res_d, sizeof(double)));
    switch (rounding) {
        case round_down:
            cuda_operation_double_rd <<< 1, 1 >>> (first_value, second_value, operation, res_d);
            break;
        case round_up:
            cuda_operation_double_ru <<< 1, 1 >>> (first_value, second_value, operation, res_d);
            break;
        case round_to_nearest:
            
            break;
        case round_toward_zero:
            
            break;
    }
    CHECK_CUDA_ERROR
    SAFE_CALL(cudaMemcpy(res_h, res_d, sizeof(double), cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaFree(res_d));

    return * res_h;
}

void ariadne_cuda::_ifma(std::vector < int* >& r_index_vector, std::vector < double >& r_value_vecto, int r_size, 
    std::vector < int* >& y_index_vector, std::vector < double >& y_value_vecto, int y_size) {
    return;
}
