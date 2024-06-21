#pragma once
// #include <iostream>
#include <stdio.h>
#include <omp.h>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>

#include <stdlib.h>
#include <sys/time.h>
#define length 16384

double oper( double x,  double y)  { 
            return x + y;
        }

int main() {
    // int length = 16;
    double *a = new double[length*length];
    // double b[length];
    // double b[length];
    // double c[length];
    for(int i = 0; i < length*length; ++i){
        a[i] = (double)(rand() % 100);
        // b[i] = a[i];
    }
    double sum = 0;
    struct timeval start, end;
    // op<double> oper;
    gettimeofday(&start, NULL);
    // #pragma omp declare reduction (g_oper: double : omp_out = oper(omp_out, omp_in)) \
    //     initializer(omp_priv=0) 
    #pragma omp parallel for simd reduction(+:sum) //collapse(2)  
    for(int i = 0; i < length; ++i){
        for(int j = 0; j < length; ++j)
            sum = oper(sum, a[i*length + j]);
    }
    gettimeofday(&end, NULL);
    double seconds = 1000 * (end.tv_sec - start.tv_sec) + 1.0e-3 * (end.tv_usec - start.tv_usec);
    printf("cpu time:%.6f, sum:%f\n", seconds, sum);
    // std::cout << "openmp:" <<seconds << " " << sum << std::endl;
    sum = 0;
    gettimeofday(&start, NULL);
    // #pragma omp parallel for simd reduction(+:sum)  
    for(int i = 0; i < length; ++i){
        for(int j = 0; j < length; ++j)
            sum = oper(sum, a[i*length + j]);
    }
    gettimeofday(&end, NULL);
    seconds = 1000 * (end.tv_sec - start.tv_sec) + 1.0e-3 * (end.tv_usec - start.tv_usec);
    printf("cpu seq time:%.6f, sum:%f\n", seconds, sum);
    // std::cout <<"seq:" <<seconds << std::endl;
    // cudaEvent_t start1, stop1;
    // op<double> oper;

    // cudaMalloc((void**)&d_data, sizeof(double)*length);
	// cudaEventCreate(&start1);
	// cudaEventCreate(&stop1);
	// cudaEventRecord(start1, 0);
    // sum = 0;

    // gettimeofday(&start, NULL);
    // #pragma omp target data  map(to:a[0:length],b[0:length]) map(tofrom:sum)
    // {
    //         #pragma omp target teams distribute parallel for simd reduction(+:sum) 
    //         for(int i = 0; i < length; ++i){
    //             sum += a[i]*b[i];//oper(sum,a[i]);
    //         }
    // }   

    // #pragma omp target update from(sum)
    // gettimeofday(&end, NULL);
    // seconds = 1000 * (end.tv_sec - start.tv_sec) + 1.0e-3 * (end.tv_usec - start.tv_usec);
    // printf("gpu time:%.6f, sum:%f\n", seconds, sum);
    delete []a;
    return 0;

}