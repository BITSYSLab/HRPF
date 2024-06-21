#pragma once
#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include <ctime>


#include <thrust/system/cuda/execution_policy.h>

#include <iostream>

// DSIZE determines duration of H2D and D2H transfers
// #define DSIZE (1048576*8)
// // SSIZE,LSIZE determine duration of kernel launched by thrust
// #define SSIZE (1024*512)
// #define LSIZE 1
// // KSIZE determines size of thrust kernels (number of threads per block)
// #define KSIZE 64
// #define TV1 1
// #define TV2 2

// typedef int mytype;
// typedef thrust::host_vector<mytype, thrust::cuda::experimental::pinned_allocator<mytype> > pinnedVector;



// #include "cuMerge.h"
void test(double* data, int len) {

	//thrust::sort(data, data+len);
	thrust::sort(thrust::device_pointer_cast(data), thrust::device_pointer_cast(data+len));
}


int main() {
	int n = 1024;
	double* data = new double[n];
	double* data_d;
	cudaEvent_t start, stop;
	//struct timespec start, stop;
	cudaStream_t stream;
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaMalloc((void**)&data_d, n*sizeof(double));
	for(int i = 0; i < n; ++i){
	
		data[i] = rand() % 10000;
	}
	cudaEventRecord(start, 0);
	//cudaMemcpy(data_d, data, n*sizeof(double), cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);
	//clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	cudaMemcpy(data_d, data, n*sizeof(double), cudaMemcpyHostToDevice);
	//thrust::sort(thrust::device_pointer_cast(data_d), thrust::device_pointer_cast(data_d + n));
	//cudaEventSynchronize(stop);
	//thrust::sort(data, data+n);
	//test(data_d, n);
	thrust::sort(thrust::cuda::par.on(stream), thrust::device_pointer_cast(data_d), thrust::device_pointer_cast(data_d+n));
	// gsort(data_d, n);
	// cudaMemcpy(data, data_d, n*sizeof(double), cudaMemcpyDeviceToHost);
	//clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
	//cudaMemcpy(data, data_d, n*sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float times;
	cudaEventElapsedTime(&times, start, stop);
	//times = (stop.tv_sec-start.tv_sec)*1e3 + (stop.tv_nsec- start.tv_nsec)/1e6;
	std::cout << times << std::endl;
	// for(int i = 0; i < n; ++i){
	// 	std::cout << data[i] << " ";
	// }
	cudaEventDestroy(stop);
	cudaEventDestroy(start);
	cudaStreamDestroy(stream);
	delete data;
	cudaFree(data_d);
	return 0;

}
