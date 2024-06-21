#include <iostream>
#include <stdio.h>
#include <omp.h>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <openacc.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cmath>

#define dim 4096
int main() {
    double *a = new double[dim*dim];
    double *b = new double[dim];
    double *c = new double[dim];

    for(int i = 0; i < dim; ++i){
        for(int j = 0; j < dim; ++j) {
            a[j + i*dim] = (double)(rand() % 100);
            
            // std::cout << "("<< j << "," << i<< ")" << j + i*dim << std::endl;
        }
    }

    for(int i = 0; i < dim; ++i){
        b[i] = (double)(rand() % 100);
    }

    struct timeval start, end;
    int i,j;
    gettimeofday(&start, NULL);
    #pragma omp parallel for simd shared(a,b,c) private(j,i) 
    for( j = 0; j < dim; ++j){
        double loc = 0;
        for( i = 0; i < dim; ++i)
        {
            loc += (a[j+i *dim] * b[i]);
        }
        c[j] = loc;
        // printf("j:%d,c:%f\n", j, c[j]);
    }
    gettimeofday(&end, NULL);
    double seconds = 1000 * (end.tv_sec - start.tv_sec) + 1.0e-3 * (end.tv_usec - start.tv_usec);
    printf("cpu time:%.6f\n", seconds);

    gettimeofday(&start, NULL);
    // #pragma omp parallel for //num_threads(8)  
    for( j = 0; j < dim; ++j){
        double loc = 0;
        for( i = 0; i < dim; ++i)
        {
            loc += a[j+i *dim] * b[i];
        }
        c[j] = loc;
        // printf("j:%d,c:%f\n", j, c[j]);
    }
    gettimeofday(&end, NULL);
    seconds = 1000 * (end.tv_sec - start.tv_sec) + 1.0e-3 * (end.tv_usec - start.tv_usec);
    printf("cpu seq time:%.6f\n", seconds);

    delete [] a;
    delete [] b;
    delete [] c;
}