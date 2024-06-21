#pragma once
#include "cuAdd.h"
#include <cstdio>
#include <iostream>

#define TILE_SIZE c

__global__ void sumMatrixKernel(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;

    int idx_a = ix + iy * lda;
    int idx_b = ix + iy * ldb;
    int idx_c = ix + iy * ldc;
    if(ix < dim && iy < dim)
    MatC[idx_c] = MatA[idx_a] + MatB[idx_b];
}


__global__ void MatrixSumKernel(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, const double p_one)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;

	int idx_a = ix + iy * lda;
	int idx_b = ix + iy * ldb;
	int idx_c = ix + iy * ldc;

	if(ix < dim && iy < dim)
		MatC[idx_c] += MatA[idx_a] + MatB[idx_b] * p_one;
}


__global__ void subMatrixKernel(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;

    int idx_a = ix + iy * lda;
    int idx_b = ix + iy * ldb;
    int idx_c = ix + iy * ldc;
    if(ix < dim && iy < dim)
        MatC[idx_c] = MatA[idx_a] - MatB[idx_b];
}

__global__ void matmul_kernel(_TYPE* a, _TYPE* b, _TYPE* c, int dim, int lda, int ldb, int ldc){
    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for(int i=thread_y; i < dim; i = i + stride_y){
        for(int j=thread_x; j < dim;j = j + stride_x){
            //printf("%d, %d", i, j);
            _TYPE value = 0;
            for(int k=0; k< dim; k++){
                value = value + a[i*lda + k]*b[k*ldb + j];
            }
            c[i*ldc + j] = value;
        }
    }
}

void sumMatrixInplace(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, const double p_one){
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks((dim-1)/threadsPerBlock.x+1, (dim-1)/threadsPerBlock.y+1);
	
	MatrixSumKernel<<<numBlocks, threadsPerBlock>>>(MatA, MatB, MatC, dim, lda, ldb, ldc, p_one);	

}

void sumMatrix(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, cudaStream_t stream)
{
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((dim-1) / threadsPerBlock.x +1 , (dim-1) / threadsPerBlock.y + 1);
	//printf("%d,%d", numBlocks,threadsPerBlock);
	//std::cout << grid.x << grid.y << std::endl;
    sumMatrixKernel<<<numBlocks, threadsPerBlock, 0, stream>>> (MatA, MatB, MatC, dim, lda, ldb, ldc);
}
void subMatrix(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, cudaStream_t stream)
{
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((dim-1) / threadsPerBlock.x +1 , (dim-1) / threadsPerBlock.y + 1);

    subMatrixKernel<<<numBlocks, threadsPerBlock,0, stream>>>(MatA, MatB, MatC, dim, lda, ldb, ldc);
}


// __global__ void gpu_Matrix_Mul_shared(double *d_a, double *d_b, double *d_c, const int size, const int ld)
// {
//     int row, col;
//     //Defining Shared Memory
//     __shared__ double shared_a[TILE_SIZE][TILE_SIZE];
//     __shared__ double shared_b[TILE_SIZE][TILE_SIZE];
//     col = TILE_SIZE * blockIdx.x + threadIdx.x;
//     row = TILE_SIZE * blockIdx.y + threadIdx.y;

//     for (int i = 0; i < size / TILE_SIZE; i++)
//     {
//         shared_a[threadIdx.y][threadIdx.x] = d_a[row* ld + (i*TILE_SIZE + threadIdx.x)];
//         shared_b[threadIdx.y][threadIdx.x] = d_b[(i*TILE_SIZE + threadIdx.y) * ld + col];
//         __syncthreads();
//         for (int j = 0; j < TILE_SIZE; j++)
//         d_c[row*ld + col] += shared_a[threadIdx.y][j] * shared_b[j][threadIdx.x];
//         __syncthreads();
//     }
// }

void gemm(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, cudaStream_t stream)
{
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((dim-1) / threadsPerBlock.x +1 , (dim-1) / threadsPerBlock.y + 1);

    matmul_kernel<<<numBlocks, threadsPerBlock,0, stream>>> (MatA, MatB, MatC, dim, lda, ldb, ldc);

    // dim3 dimGrid(dim / TILE_SIZE, dim / TILE_SIZE, 1);
    // dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    // gpu_Matrix_Mul_shared << <dimGrid, dimBlock, 2048, stream>> > (MatA, MatB, MatC, dim, lda);
}