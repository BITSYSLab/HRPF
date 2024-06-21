#pragma once
#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <sys/time.h>

int main(int argc, char **argv) {
    int length = std::atoi(argv[1]);
    double* a = new double[length];
    double* b = new double[length];
    double* c = new double[length];
    for(int i = 0; i < length; ++i){
        a[i] = (double)(rand() % 100);
        b[i] = (double)(rand() % 100);
    }
    struct timeval start, end;
    // gettimeofday(&start, NULL);
    // #pragma omp parallel for
    // for(int i = 0; i < length; ++i){
    //     c[i] = a[i] + b[i];
    // }
    // gettimeofday(&end, NULL);
    // double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    // std::cout << seconds << std::endl;
    gettimeofday(&start, NULL);
    // #pragma omp parallel for simd
    for(int i = 0; i < length; ++i){
        c[i] = a[i] + b[i];
    }
    gettimeofday(&end, NULL);
    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    std::cout << seconds << std::endl;

    delete []a;
    delete []b;
    delete []c;
    return 0;

}