/*
 * @Author: your name
 * @Date: 2022-03-03 10:09:18
 * @LastEditTime: 2022-03-03 14:28:05
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_12\algorithm\parallel_for\parallel_for_harness.cpp
 */
#pragma once
#include <stdio.h>
#include <sys/time.h>
#include <string>

#include "parallel_for_zb.h"
#include "tool/initializer.h"
#include "framework/framework.h"
#include "tool/helper.h"
// #include "parallel_for_harness.hpp"
#include <omp.h>

struct UserData_t : public Basedata_t{
public:
    UserData_t(std::vector<ArrayList*> buf
        ) : buffer(buf){
        }

public:
    std::vector<ArrayList*> buffer;
};

void cfor_func(Basedata_t* data){
    // std::cout << "cpu exec" << std::endl;
    auto d = (loopData_t*)data;
    auto a = ((UserData_t*)(d->buffer))->buffer[0]->get_cdata();
    auto b = ((UserData_t*)(d->buffer))->buffer[1]->get_cdata();
    auto c = ((UserData_t*)(d->buffer))->buffer[2]->get_cdata();

    size_t s = d->start;
    size_t e = d->end;
    #pragma omp parallel for
    for(int i = s; i < e; ++i){
        c[i] = a[i] + b[i];
    }
}

__global__ void kernel(size_t s, size_t e, size_t chunk, double* a, double* b, double* c){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = s + tid * chunk;
    int end = start+chunk < e ? start + chunk : e;

    for(int i = start; i < end; ++i){
        c[i] = a[i] + b[i];
    }
}

void gfor_func(Basedata_t* data){
    std::cout << "gpu exec" << std::endl;
    auto d = (loopData_t*)data;
    auto a = ((UserData_t*)(d->buffer))->buffer[0]->get_gdata();
    auto b = ((UserData_t*)(d->buffer))->buffer[1]->get_gdata();
    auto c = ((UserData_t*)(d->buffer))->buffer[2]->get_gdata();

    size_t s = d->start;
    size_t e = d->end;

    int blocks_required = 1;
    int threads_per_block = 1024;
    int chunk_size = 1;
    int size = e - s;
    if(size % (threads_per_block * chunk_size)) {
        blocks_required = size / (threads_per_block * chunk_size) + 1;
    }
    else {
        blocks_required = size / (threads_per_block * chunk_size);
    }
    cudaStream_t stream_ = stream();
    kernel<<<blocks_required, threads_per_block, 0, stream_>>>(s, e, chunk_size, a, b, c);
}

int main(int argc, char **argv){
    Framework::init();
    std::size_t length = std::atoi(argv[1]);
    ArrayList* data1 = new ArrayList(length);
    ArrayList* data2 = new ArrayList(length);
    ArrayList* data3 = new ArrayList(length);
    initialize(data1, length);
    initialize(data2, length);
    initialize(data3, length);

    UserData_t* user = new UserData_t({data1, data2, data3});
    // auto loop = new loopData_t(0, length, user);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    parallel_for(new loopData_t(0, length, user), cfor_func, gfor_func);
    // gfor_func(loop);
    gettimeofday(&end, NULL);
    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    std::cout << seconds << std::endl;
    auto res = data3->get_cdata();
    for(int i = 0; i < length; ++i) {
        // if(i && i % 16 == 0) std::cout << std::endl;
        std::cout <<(data1->get_cdata())[i] << " " <<(data2->get_cdata())[i] << " "<< res[i] << std::endl;
    }

    delete user;
    delete data1;
    delete data2;
    delete data3;
    return 0;
}
