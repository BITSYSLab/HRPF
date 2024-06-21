/*
 * @Author: your name
 * @Date: 2022-02-16 18:12:50
 * @LastEditTime: 2022-02-16 18:22:45
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_NEW_VERSION\HRPA_NEW\algorithm\strassen_problem\test.cpp
 */
#include "cuAdd.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cstdio>
#include <sys/time.h>
#include <thread>

void print(_TYPE* a, int dim) {
    for(int i = 0; i < dim*dim; ++i){
        std::cout << a[i] << "  " ;
        if(i && i % dim == 0) std::cout << std::endl;
    }
	std::cout << std::endl;
}
int main(){
    int dim = 16384;
    _TYPE* a, *b, *c;
    a = new _TYPE[dim*dim];
    b = new _TYPE[dim*dim];
    c = new _TYPE[dim*dim];

    _TYPE* a_d, *b_d, *c_d;
    cudaMalloc((void**)&a_d, dim*dim*sizeof(_TYPE));
    cudaMalloc((void**)&b_d, dim*dim*sizeof(_TYPE));
    cudaMalloc((void**)&c_d, dim*dim*sizeof(_TYPE));
   
    for(int i=0; i<dim*dim; i++){
        a[i] = 1;
            b[i] =1;
    }
	int index = (1%2)*dim/2 + 1/2*dim/2*dim/2; 
    cudaStream_t stream;
	cudaStreamCreate(&stream);
     struct timeval start, end;
    gettimeofday(&start, NULL);
	cudaMemcpy(a_d, a, dim*dim*sizeof(_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, dim*dim*sizeof(_TYPE), cudaMemcpyHostToDevice);
   
    gemm(a_d, b_d, c_d, dim, dim, dim, dim, stream);
	// sumMatrix(a_d, b_d, c_d, dim/2, dim, dim,dim, NULL);
    cudaStreamSynchronize(stream);
    gettimeofday(&end, NULL);
    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
//    print(c, dim);
	cudaMemcpy(c, c_d, dim*dim*sizeof(_TYPE), cudaMemcpyDeviceToHost);
	// for(int i = 0; i < dim; ++i){
	// 	for(int j = 0; j < dim; ++j){
	// 		std::cout << c[i+dim*j] << "  ";
		
	// 	}
	// 	std::cout << std::endl;
	
	// }
	std::cout << seconds << std::endl;
	cudaStreamDestroy(stream);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    delete a;
    delete b;
    delete c;
    return 0;
}
