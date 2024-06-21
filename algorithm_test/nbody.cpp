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
double dt;
int main(int argc, char **argv) {
    dim = std::atoi(argv[1]);
    double *x1 = new double[dim];
    double *x2 = new double[dim];
    double *x3 = new double[dim];
    double *mass = new double[dim];
    double *v1 = new double[dim];
    double *v2 = new double[dim];
    double *v3 = new double[dim];

    for(int i = 0; i < dim; ++i){
        x1[i] = (double)(rand() % 100);
        x2[i] = (double)(rand() % 100);
        x3[i] = (double)(rand() % 100);
        mass[i] = (double)(rand() % 100);
        v1[i] = (double)(rand() % 100);
        v2[i] = (double)(rand() % 100);
        v3[i] = (double)(rand() % 100);
    }
    dt = (double)(rand() % 10);
    struct timeval start, end;
    int i,j;
    gettimeofday(&start, NULL);
    // #pragma omp parallel for simd shared(x1,x2,x3,v1,v2,v3,mass) private(j,i) 
    for(int i = 0; i < dim; ++i){
        double Fx = 0; double Fy = 0; double Fz = 0;
        for(int j = 0; j < dim; ++j) {
            double dx = x1[j] - x1[i];
            double dy = x2[j] - x2[i];
            double dz = x3[j] - x3[i];
            double dst = dx*dx + dy*dy + dz*dz + mass[i];
            double invDist = 1.0 / sqrt(dst);
            double invDist3 = pow(invDist, 3);
            Fx += dx*invDist3; Fy += dy*invDist3; Fz += dz*invDist3;
        }
        v1[i] += dt * Fx; v2[i] += dt*Fy; v3[i] += dt*Fz;
        
    }
    // #pragma omp parallel for simd shared(x1,x2,x3,v1,v2,v3) 
    // for(int i = 0; i < dim; ++i)
    //  {   x1[i] += dt * v1[i]; x2[i] += dt * v2[i]; x3[i] += dt * v3[i];}
    gettimeofday(&end, NULL);
    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    printf("%.6f\n", seconds);

    // gettimeofday(&start, NULL);
    // for(int i = 0; i < dim; ++i){
    //     double Fx = 0; double Fy = 0; double Fz = 0;
    //     for(int j = 0; j < dim; ++j) {
    //         double dx = x1[j] - x1[i];
    //         double dy = x2[j] - x2[i];
    //         double dz = x3[j] - x3[i];
    //         double dst = dx*dx + dy*dy + dz*dz + mass[i];
    //         double invDist = 1.0 / sqrt(dst);
    //         double invDist3 = pow(invDist, 3);
    //         Fx += dx*invDist3; Fy += dy*invDist3; Fz += dz*invDist3;
    //     }
    //     v1[i] += dt * Fx; v2[i] += dt*Fy; v3[i] += dt*Fz;
    // }
    // // for(int i = 0; i < dim; ++i)
    // //      {   x1[i] += dt * v1[i]; x2[i] += dt * v2[i]; x3[i] += dt * v3[i];}
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
    delete [] x1;
    delete [] x2;
    delete [] x3;
    delete [] v1;
    delete [] v2;
    delete [] v3;
    delete [] mass;
    
}