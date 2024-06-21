#include <iostream>
#include <stdio.h>
#include <omp.h>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <openacc.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cmath>

static int dim ;

int main(int argc, char **argv) {
    dim = std::atoi(argv[1]);
    double *a = new double[dim*dim];
    double *b = new double[dim*dim];
    double *c = new double[dim*dim];

    for(int i = 0; i < dim; ++i){
        for(int j = 0; j < dim; ++j) {
            a[j + i*dim] = (double)(1);
            b[j + i*dim] = (double)(1);
            // std::cout << "("<< j << "," << i<< ")" << j + i*dim << std::endl;
        }
    }

    struct timeval start, end;
    int i,j;
    gettimeofday(&start, NULL);
    #pragma omp parallel for simd shared(a,b,c) private(j,i) 
    for( j = 0; j < dim; ++j){
        for( i = 0; i < dim; ++i)
        {
            c[i+j *dim] = a[i+j *dim] * b[i+j *dim];
        }
    }
    gettimeofday(&end, NULL);
    double seconds = 1000 * (end.tv_sec - start.tv_sec) + 1.0e-3 * (end.tv_usec - start.tv_usec);
    printf("%.6f\n", seconds);

    // gettimeofday(&start, NULL);
    // // #pragma omp parallel for //num_threads(8)  
    // for(int i = 0; i < dim; ++i){
    //     for(int j = 0; j < dim; ++j)
    //     {
    //         c[j+i *dim] = a[j+i *dim] * b[j+i *dim];
    //     }
    // }
    // gettimeofday(&end, NULL);
    // seconds = 1000 * (end.tv_sec - start.tv_sec) + 1.0e-3 * (end.tv_usec - start.tv_usec);
    // printf("cpu seq time:%.6f\n", seconds);

    // gettimeofday(&start, NULL);
    // #pragma omp target data map(from:c[0:dim*dim]) map(to:a[0:dim*dim], b[0:dim*dim])
    // {
    //         #pragma omp target teams distribute parallel for simd 
    //         for(int i = 0; i < dim; ++i){
    //             for(int j = i; j <= dim; ++j)
    //             {
    //                 c[j+i *dim] = a[j+i *dim] + b[j+i *dim];
    //             }
    //         }
    // }   
    // gettimeofday(&end, NULL);
    // seconds = 1000 * (end.tv_sec - start.tv_sec) + 1.0e-3 * (end.tv_usec - start.tv_usec);
    // printf("gpu time:%.6f\n", seconds);
    delete [] a;
    delete [] b;
    delete [] c;
}