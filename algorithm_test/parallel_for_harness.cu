/*
 * @Author: your name
 * @Date: 2022-03-03 10:09:18
 * @LastEditTime: 2022-03-03 14:28:05
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_12\algorithm\parallel_for\parallel_for_harness.cpp
 */
#pragma once

#include "parallel_for_harness.hpp"
#include <cstdio>
#define PI 3.14159
__global__ void kernel(size_t s, size_t e, size_t chunk, double* a, double* b, double* c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = s + tid * chunk;
    int end = start+chunk < e ? start + chunk : e;

    for(int i = start; i < end; ++i){
        c[i] = a[i] + b[i];
    }
}

__global__ void kernel_pro(size_t s, size_t e, size_t chunk, double* a, double* b, double* c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = s + tid * chunk;
    int end = start+chunk < e ? start + chunk : e;

    for(int i = start; i < end; ++i){
        c[i] = a[i] * b[i];
    }
}

__global__ void kernel(size_t s, size_t e, size_t chunk, 
    double* x1, double* x2, double* x3,
    double* v1, double* v2, double* v3,
    double* mass, double dt, double length) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = s + tid * chunk;
    int end = start+chunk < e ? start + chunk : e;

    for(int i = start; i < end; ++i){
        double Fx = 0; double Fy = 0; double Fz = 0;
        for(int j = 0; j < length; ++j) {
            double dx = x1[j] - x1[i];
            double dy = x2[j] - x2[i];
            double dz = x3[j] - x3[i];
            double dst = dx*dx + dy*dy + dz*dz + mass[i];
            double invDist = rsqrt(dst);
            double invDist3 = pow(invDist, 3);
            Fx += dx*invDist3; Fy += dy*invDist3; Fz += dz*invDist3;
        }
        v1[i] += dt * Fx; v2[i] += dt*Fy; v3[i] += dt*Fz;
    }
}

__global__ void kernel_2DFT(size_t s_i, size_t e_i, size_t s_j, size_t e_j, size_t chunk, 
    double* ar, double* ai, double* tr,
    double* ti, size_t length) {
    printf("enter gpu....\n");
    int tid = threadIdx.x + blockIdx.x * blockDim.x;//printf("%d\n", tid);
    int start_i = s_i + tid * chunk;
    int end_i = start_i + chunk < e_i ? start_i + chunk : e_i;
    if(tid == 0)
        printf("s:%d, e:%d\n", start_i, end_i);
    for(int i = start_i; i < end_i; ++i){
        tr[i-s_i] = 0; double wnr = 0;
        ti[i-s_i] = 0; double wni = 0;
        for(int j = s_j; j < e_j; ++j) {
           wnr = cos(2.0 * PI / length * j * i);
           wni = sin(2.0 * PI / length * j * i);
           tr[i-s_i] += (ar[j] * wnr - ai[j] * wni);
           ti[i-s_i] += (ar[j] * wni + ai[j] * wnr);
        }
        // printf("%.3f,%.3f\n", tr[i], ar[i]); 
    }

}


__global__ void kernel_2D(size_t s_i, size_t e_i, size_t s_j, size_t e_j,
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* b, double* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // int start_i = s_i + tid * chunk;
    // int end_i = start_i + chunk < e_i ? start_i + chunk : e_i;
    int start_j = s_j + tid * chunk;
    int end_j = start_j + chunk < e_j ? start_j + chunk : e_j;

    for(int j = start_j; j < end_j; ++j){
        for(int i = s_i; i < e_i; ++i) {
            c[i + j * ldc] = a[i + j * lda] + b[i + j * ldb];
        }
    }  
}

__global__ void kernel_2D_pro(size_t s_i, size_t e_i, size_t s_j, size_t e_j,
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* b, double* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // int start_i = s_i + tid * chunk;
    // int end_i = start_i + chunk < e_i ? start_i + chunk : e_i;
    int start_j = s_j + tid * chunk;
    int end_j = start_j + chunk < e_j ? start_j + chunk : e_j;

    for(int j = start_j; j < end_j; ++j){
        for(int i = s_i; i < e_i; ++i) {
            c[i + j * ldc] = a[i + j * lda] * b[i + j * ldb];
        }
    }  
}

__global__ void kernel_2DMv(size_t s_i, size_t e_i, size_t s_j, size_t e_j,
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* b, double* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int start_i = s_i + tid * chunk;
    int end_i = start_i + chunk < e_i ? start_i + chunk : e_i;

    for(int i = start_i; i < end_i; ++i){
        double loc = 0.0;
        for(int j = s_j; j < e_j; ++j) {
            loc += a[i + j * lda] * b[j];
        }
        c[i] = loc;
    }  
}

__global__ void kernel_2DKNN(size_t s_i, size_t e_i, size_t s_j, size_t e_j,
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* b, double* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int start_i = s_i + tid * chunk;
    int end_i = start_i + chunk < e_i ? start_i + chunk : e_i;

    for(int i = start_i; i < end_i; ++i){
        double loc = 0.0;
        for(int j = s_j; j < e_j; ++j) {
            loc += (a[i + j * lda] - b[j])*(a[i + j * lda] - b[j]);
        }
        c[i] = loc;
    }  
}

__global__ void kernel_2DKMS(size_t s_i, size_t e_i, size_t s_j, size_t e_j,
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* cent0, double* cent1,
    double* cent2, double* cent3, double* dist, double* index) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double* cent[4] = {cent0, cent1, cent2, cent3};
    int start_i = s_i + tid * chunk;
    int end_i = start_i + chunk < e_i ? start_i + chunk : e_i;

    for(int i = start_i; i < end_i; ++i){
        int minIdx = -1;
        double minDst = INT_MAX;
                
        for(int j = 0; j < 4; ++j){
            double sum = 0;
            for(int c = s_j; c < e_j; ++c){
                sum += (a[i + c*lda] - cent[j][c])*(a[i + c*lda] - cent[j][c]);
            }
            if(sum < minDst) {
                minDst = sum;
                minIdx = j;
            }
        }

        dist[i] = minDst;
        index[i] = minIdx;
    }  
}

__global__ void kernelR_2D(size_t s_i, size_t e_i, size_t s_j, size_t e_j, 
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* b, double* c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;    
    int start_j = s_j + tid * chunk;
    int end_j = start_j + chunk < e_j ? start_j + chunk : e_j;
    int end_i = e_i;
    
    for(int j = start_j; j < end_j; ++j) {
        for(int i = 0; i <= end_i + j; ++i) {
            c[i + j * ldc] = a[i + j * lda] + b[i + j * ldb];
        }
    }
}

__global__ void kernelI_2D(size_t s_i, size_t e_i, size_t s_j, size_t e_j, 
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* b, double* c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;    
    int start_j = s_j + tid * chunk;
    int end_j = start_j + chunk < e_j ? start_j + chunk : e_j;
    int start_i = s_i;
    
    for(int j = start_j; j < end_j; ++j) {
        for(int i = start_i; i < e_i; ++i) {
            c[i + j * ldc] = a[i + j * lda] + b[i + j * ldb];
        }
    }
}
