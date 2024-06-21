/*
 * @Author: your name
 * @Date: 2022-03-03 11:17:40
 * @LastEditTime: 2022-03-03 14:20:45
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_12\algorithm\parallel_for\loopfunc.h
 */
#pragma once
// #include "parallel_for_problem.h"

__global__ void kernel(size_t s, size_t e, size_t chunk, double* a, double* b, double* c);

__global__ void kernel(size_t s, size_t e, size_t chunk, 
    double* x1, double* x2, double* x3,
    double* v1, double* v2, double* v3,
    double* mass, double dt, double length);

__global__ void kernel_2D(size_t s_i, size_t e_i, size_t s_j, size_t e_j, 
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* b, double* c);

__global__ void kernelR_2D(size_t s_i, size_t e_i, size_t s_j, size_t e_j, 
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* b, double* c);

__global__ void kernelI_2D(size_t s_i, size_t e_i, size_t s_j, size_t e_j, 
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* b, double* c);

__global__ void kernel_2D_pro(size_t s_i, size_t e_i, size_t s_j, size_t e_j, 
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* b, double* c);

__global__ void kernel_2DMv(size_t s_i, size_t e_i, size_t s_j, size_t e_j, 
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* b, double* c);

__global__ void kernel_2DKNN(size_t s_i, size_t e_i, size_t s_j, size_t e_j,
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* b, double* c);

__global__ void kernel_2DKMS(size_t s_i, size_t e_i, size_t s_j, size_t e_j,
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* cent0, double* cent1,
    double* cent2, double* cent3, double* dist, double* index);

__global__ void kernel_2DFT(size_t s_i, size_t e_i, size_t s_j, size_t e_j, size_t chunk, 
    double* ar, double* ai, double* tr,
    double* ti, size_t length);