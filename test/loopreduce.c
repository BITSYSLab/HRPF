#pragma once
// #include <iostream>
#include <stdio.h>
#include <omp.h>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>

#include <stdlib.h>
#include <sys/time.h>
#define length 8192


// template <typename T>
// struct op
// {
//     __host__ __device__
//         T operator()(const T& x, const T& y) const { 
//             return x + y;
//         }
// };

 
double oper( double x,  double y)  { 
            return x + y;
        }

int main() {
    // int length = 16;
    double a[length];
    double b[length];
    // double b[length];
    // double c[length];
    for(int i = 0; i < length; ++i){
        a[i] = (double)(rand() % 100);
        b[i] = a[i];
    }
    double sum = 0;
    struct timeval start, end;
    // op<double> oper;
    gettimeofday(&start, NULL);
    // #pragma omp declare reduction (g_oper: double : omp_out = oper(omp_out, omp_in)) \
    //     initializer(omp_priv=0) 
    #pragma omp parallel for simd reduction(+:sum)  
    for(int i = 0; i < length; ++i){
        sum = oper(sum, a[i]*b[i]);
    }
    gettimeofday(&end, NULL);
    double seconds = 1000 * (end.tv_sec - start.tv_sec) + 1.0e-3 * (end.tv_usec - start.tv_usec);
    printf("cpu time:%.6f, sum:%f\n", seconds, sum);
    // std::cout << "openmp:" <<seconds << " " << sum << std::endl;
    sum = 0;
    gettimeofday(&start, NULL);
    // #pragma omp parallel for simd reduction(+:sum)  
    for(int i = 0; i < length; ++i){
        sum = oper(sum, a[i]*b[i]);
    }
    gettimeofday(&end, NULL);
    seconds = 1000 * (end.tv_sec - start.tv_sec) + 1.0e-3 * (end.tv_usec - start.tv_usec);
    printf("cpu seq time:%.6f, sum:%f\n", seconds, sum);
    // std::cout <<"seq:" <<seconds << std::endl;
    // cudaEvent_t start1, stop1;
    // op<double> oper;
    double *d_data;
    // cudaMalloc((void**)&d_data, sizeof(double)*length);
	// cudaEventCreate(&start1);
	// cudaEventCreate(&stop1);
	// cudaEventRecord(start1, 0);
    sum = 0;
    // cudaMemcpy(d_data, a, sizeof(double)*length, cudaMemcpyHostToDevice);
    // #pragma omp declare  reduction (oper: double : omp_out = oper(omp_out, omp_in)) \
    //     initializer(omp_priv=sum) 
    // #pragma omp target teams distribute parallel for reduction(+:sum) map(to:a[0:length]) map(tofrom:sum)
    //     for(int i = 0 ; i < length; i++) sum += a[i];
    // #pragma omp target teams distribute parallel for simd reduction(oper:sum) map(tofrom:sum)
    //      for(int i = 0 ; i < length; i++) sum += i%11;
    //#pragma omp target device(1)
    gettimeofday(&start, NULL);
    #pragma omp target data  map(to:a[0:length],b[0:length]) map(tofrom:sum)
    {
            #pragma omp target teams distribute parallel for simd reduction(+:sum) 
            for(int i = 0; i < length; ++i){
                sum += a[i]*b[i];//oper(sum,a[i]);
            }
    }   

    // #pragma omp target update from(sum)
    gettimeofday(&end, NULL);
    seconds = 1000 * (end.tv_sec - start.tv_sec) + 1.0e-3 * (end.tv_usec - start.tv_usec);
    printf("gpu time:%.6f, sum:%f\n", seconds, sum);
        //#pragma omp target exit data map(from: sum)
    //  #pragma omp single depend(sum) {
    //     printf("%f\n", sum);
    // }
    // cudaEventRecord(stop1, 0);
	// cudaEventSynchronize(stop1);
	// float elapsedTime;
	// cudaEventElapsedTime(&elapsedTime, start1, stop1);
   
    // printf("TIME TAKEN(omp Parallel cpu & GPU): %fms%d\n", elapsedTime,sum);
	// cudaEventDestroy(start1);
	// cudaEventDestroy(stop1);	
    // cudaFree(d_data);	
    return 0;

}