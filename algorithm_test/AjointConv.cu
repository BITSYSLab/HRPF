/*
 * @Author: your name
 * @Date: 2022-03-03 10:09:18
 * @LastEditTime: 2022-03-12 14:48:19
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_12\algorithm\parallel_for\parallel_for_harness.cpp
 */
#pragma once
#include <stdio.h>
#include <sys/time.h>
#include <string>

#include "algorithm/parallel_for_zero/parallel_for_zb.h"
#include "tool/initializer.h"
#include "framework/framework.h"
#include "tool/helper.h"
#include <omp.h>
// #include <cmath>
#include <fstream>

static int length;
#define PI 3.14159

struct UserData_t : public Basedata_t{
public:
    UserData_t(std::vector<ArrayList*> buf
        ) : buffer(buf){
        }

public:
    std::vector<ArrayList*> buffer;
};

void cfor_func(Basedata_t* data){
    // std::cout << "here" << std::endl;
    auto d = (loopData_t*)data;
    auto a = ((UserData_t*)(d->buffer))->buffer[0]->get_cdata();
    auto b = ((UserData_t*)(d->buffer))->buffer[1]->get_cdata();
    auto c = ((UserData_t*)(d->buffer))->buffer[2]->get_cdata();
    size_t s_i = d->start;
    size_t e_i = d->end;
    size_t e_j = length*length;
    // std::cout << s_i << e_i << std::endl;
    #pragma omp parallel for simd
    for(int i = s_i; i < e_i ; ++i){
        double cur_c = 0.0;
        for(int j = i; j < e_j; ++j) {
          cur_c += 5.5*b[j]*a[j-i];
        } 
        c[i] = cur_c;
    }
}

__global__ void kernel_conv(size_t s_i, size_t e_i, size_t s_j, size_t e_j, size_t chunk, 
    double* a, double* b, double* c, size_t length) {
    // printf("enter gpu....\n");
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = s_i + tid * chunk;
    int end = start+chunk < e_i ? start + chunk : e_i;
    //printf("%d\n", tid);
    // int start_i = s_i; //+ tid * chunk;
    // int end_i = e_i;//start_i + chunk < e_i ? start_i + chunk : e_i;
    // if(tid == 0)
    //     printf("s:%d, e:%d\n", start_i, end_i);
    
    for(int i = start; i < end; ++i){
        double cur_c = 0.0;
        for(int j = i; j < e_j; ++j) {
            cur_c += 5.5*b[j]*a[j-i];
        }
        c[i] = cur_c;
    // printf("%.3f,%.3f\n", tr[i], ar[i]); 
    } 
    


}

void gfor_func(Basedata_t* data){
    // std::cout << "enet fgd" << std::endl;
    auto d = (loopData_t*)data;
    auto a = ((UserData_t*)(d->buffer))->buffer[0]->get_gdata();
    auto b = ((UserData_t*)(d->buffer))->buffer[1]->get_gdata();
    auto c = ((UserData_t*)(d->buffer))->buffer[2]->get_gdata();
    
    size_t s_i = d->start;
    size_t e_i = d->end;
    size_t s_j = 0;
    size_t e_j = length*length;
    // std::cout << s_i << e_i << "g" << std::endl;
    // std::cout << "s_i" << s_i << " " << e_i << std::endl;
    int blocks_required = 1;
    int threads_per_block = 1024;
    int chunk_size = 1;
    int size = e_i - s_i;
    if(size % (threads_per_block * chunk_size)) {
        blocks_required = size / (threads_per_block * chunk_size) + 1;
    }
    else {
        blocks_required = size / (threads_per_block * chunk_size);
    }
    // std::cout << blocks_required << std::endl;
    cudaStream_t stream_ = stream();//, 0, stream_
    cudaError_t error;
    kernel_conv<<<blocks_required, threads_per_block, 0, stream_>>>(s_i, e_i, s_j, e_j, 
        chunk_size, a,b,c, length);
    // double* da;

    // error = (cudaError_t)cudaGetLastError();
    // std::cout << cudaGetErrorString(error) << std::endl;
}

void loadData(double* datar, double* datai, int length) { 
    std::ifstream fin;
    fin.open("datadft.txt");
 
	if(!fin)
	{
		std::cout<<"can not open the file data.txt"<<std::endl;
		exit(1);
	}

    for(int i = 0; i < length; ++i){
        fin >> datar[i] >> datai[i];
    }
}

void print(double* datar, int length) {
    for(int i = 0; i < length; ++i){
        std::cout << datar[i] << " " ;;//<< datai[i] << std::endl;
        if(i && i % 4 == 0) std::cout << std::endl;
    }
}

int main(int argc, char **argv){
    Framework::init();
    std::size_t N = std::atoi(argv[1]);
    length = N;
    ArrayList* a = new ArrayList(length*length);
    ArrayList* b = new ArrayList(length*length);
    ArrayList* c = new ArrayList(length*length);
    auto& runtime = Runtime::get_instance();
    auto cpu = runtime.get_cpu();
    (a)->access(cpu, MemAccess::W);
    (b)->access(cpu, MemAccess::W);
    loadData(a->get_cdata(), b->get_cdata(), length*length);
    
    UserData_t* user = new UserData_t({a,b,c}); 
    struct timeval start, end;
    gettimeofday(&start, NULL);
    parallel_for(new loopData_t(0, length*length, user), cfor_func, gfor_func);
    gettimeofday(&end, NULL);
    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    std::cout << seconds << std::endl;
    // cudaDeviceSynchronize();
    // datar->copy_from(datar->get_cdata(), datar->get_gdata(), Runtime::get_instance().get_cpu());
    
    auto da = c->get_cdata();

    // print(da, length*length);
    delete a;
    delete b;
    delete c;
    return 0;
}
