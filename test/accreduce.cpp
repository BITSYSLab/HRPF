#pragma once
// #include <iostream>
#include <stdio.h>
#include <omp.h>
#include <openacc.h>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
#include <stdlib.h>
#include <chrono>
#include <ctime>
#include <sys/time.h>
// #define length 1024
using namespace std::chrono;
double oper(const double& x, const double& y)  { 
            return x + y;
        }
int main() {
    int length = 1024;
    double a[length];
    // double b[length];
    // double c[length];
    for(int i = 0; i < length; ++i){
        a[i] = (double)(1);
        // b[i] = a[i];
    }
    // printf("%d\n", acc_get_num_devices(2));
    double sum = 0;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    // #pragma omp parallel for simd reduction(+:sum)  
    for(int i = 0; i < length; ++i){
        sum = oper(sum, a[i]);
    }
    gettimeofday(&end, NULL);
    double seconds = 1000 * (end.tv_sec - start.tv_sec) + 1.0e-3 * (end.tv_usec - start.tv_usec);
    printf("cpu seq time:%.6f, sum:%f\n", seconds, sum);
    sum = 0;
    gettimeofday(&start, NULL);
    // clock_t time;
    // time = clock();
    // #pragma omp declare reduction (g_oper: double : omp_out = oper(omp_out, omp_in)) \
    //     initializer(omp_priv=0) 

    #pragma acc parallel loop copyin(a[0:length]) copy(sum) reduction(+:sum)    
    for(int i = 0; i < length; ++i){
        sum += a[i];
    }
    // #pragma wait
    // time = clock() - time;
    // printf("\nTime with acc:%d ms\n", time);

    gettimeofday(&end, NULL);
     seconds = 1000 * (end.tv_sec - start.tv_sec) + 1.0e-3 * (end.tv_usec - start.tv_usec);
    printf("times:%0.6f, sum:%f\n", seconds, sum);
    // std::cout << seconds << " " << sum << std::endl;
    // cudaEvent_t start1, stop1;
	// cudaEventCreate(&start1);
	// cudaEventCreate(&stop1);
	// cudaEventRecord(start1, 0);
    // sum = 0;
    // #pragma target data map(to:a[0:length]) map(tofrom:sum)
    // {
    //     #pragma omp target teams distribute parallel for simd reduction(+:sum) 
        
    //         for(int i = 0; i < length; ++i){
    //             sum += a[i];
    //         }
        
    // }
    // cudaEventRecord(stop1, 0);
	// cudaEventSynchronize(stop1);
	// float elapsedTime;
	// cudaEventElapsedTime(&elapsedTime, start1, stop1);
    // printf("TIME TAKEN(omp Parallel cpu & GPU): %fms\n", elapsedTime);
	// cudaEventDestroy(start1);
	// cudaEventDestroy(stop1);	
    return 0;

}