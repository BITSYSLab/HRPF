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
    auto d = (loopData_t*)data;
    auto ar = ((UserData_t*)(d->buffer))->buffer[0]->get_cdata();
    auto ai = ((UserData_t*)(d->buffer))->buffer[1]->get_cdata();
    auto tr = ((UserData_t*)(d->buffer))->buffer[2]->get_cdata();
    auto ti = ((UserData_t*)(d->buffer))->buffer[3]->get_cdata();
    
    size_t s_i = d->start;
    size_t e_i = d->end;
    size_t s_j = 0;
    size_t e_j = length;
    // std::cout << s_i << s_j << e_i << e_j << std::endl;
    #pragma omp parallel for simd// private(j,i)
    for(int i = s_i; i < e_i ; ++i){
        tr[i] = 0; double wnr = 0;
        ti[i] = 0; double wni = 0;
        for(int j = s_j; j < e_j; ++j) {
           wnr = cos(2.0 * PI / length * j * (i));
           wni = sin(2.0 * PI / length * j * (i));
           tr[i] += (ar[j] * wnr - ai[j] * wni);
           ti[i] += (ar[j] * wni + ai[j] * wnr);
        } 
    }
}

__global__ void kernel_2DFT1(size_t s_i, size_t e_i, size_t s_j, size_t e_j, size_t chunk, 
    double* ar, double* ai, double* tr,
    double* ti, size_t length) {
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
        tr[i] = 0; double wnr = 0;
        ti[i] = 0; double wni = 0;
        for(int j = s_j; j < e_j; ++j) {
        wnr = cos(2.0 * PI / length * j * i);
        wni = sin(2.0 * PI / length * j * i);
        tr[i] += (ar[j] * wnr - ai[j] * wni);
        ti[i] += (ar[j] * wni + ai[j] * wnr);
        }
    // printf("%.3f,%.3f\n", tr[i], ar[i]); 
        } 
    


}

void gfor_func(Basedata_t* data){
    std::cout << "enet fgd" << std::endl;
    auto d = (loopData_t*)data;
    auto ar = ((UserData_t*)(d->buffer))->buffer[0]->get_cdata();
    auto ai = ((UserData_t*)(d->buffer))->buffer[1]->get_cdata();
    auto tr = ((UserData_t*)(d->buffer))->buffer[2]->get_cdata();
    auto ti = ((UserData_t*)(d->buffer))->buffer[3]->get_cdata();
    
    size_t s_i = d->start;
    size_t e_i = d->end;
    size_t s_j = 0;
    size_t e_j = length;
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
    kernel_2DFT1<<<blocks_required, threads_per_block, 0, stream_>>>(s_i, e_i, s_j, e_j, 
        chunk_size, ar, ai, tr, ti, length);
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

int main(int argc, char **argv){
    Framework::init();
    std::size_t N = std::atoi(argv[1]);
    length = N;
    ArrayList* datar = new ArrayList(length);
    ArrayList* datai = new ArrayList(length);
    ArrayList* tempr = new ArrayList(length);
    ArrayList* tempi = new ArrayList(length);
    // initialize(datar, length);
    // initialize(datai, length);
    auto& runtime = Runtime::get_instance();
    auto cpu = runtime.get_cpu();
    (datar)->access(cpu, MemAccess::W);
    (datai)->access(cpu, MemAccess::W);
    loadData(datar->get_cdata(), datai->get_cdata(), length);
    // auto cd = data1->get_cdata();
    // for(int i = 0; i < length; ++i){
    //     for(int j = 0; j < length; ++j){
    //         std::cout << cd[j + length*i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // datar->copy_from(datar->get_gdata(), datar->get_cdata(), Runtime::get_instance().get_gpu());
    // datai->copy_from(datai->get_gdata(), datai->get_cdata(), Runtime::get_instance().get_gpu());
    // auto da = datar->get_cdata();
    UserData_t* user = new UserData_t({datar,datai,tempr, tempi}); 
    struct timeval start, end;
    gettimeofday(&start, NULL);
    parallel_for(new loopData_t(0, length, user), cfor_func, gfor_func);
    gettimeofday(&end, NULL);
    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    std::cout << seconds << std::endl;
    // cudaDeviceSynchronize();
    // datar->copy_from(datar->get_cdata(), datar->get_gdata(), Runtime::get_instance().get_cpu());
    
    auto da = tempr->get_cdata();

    for(int i = 0; i < length; ++i){
        // for(int j = 0; j < length; ++j){
        
        std::cout << da[i] << " ";
        // }
        if(i && i % 4 == 0)
        std::cout << std::endl;
    }

    // da = tempr->get_cdata();
    // for(int i = 0; i < length; ++i){
    //     // for(int j = 0; j < length; ++j){
        
    //     std::cout << da[i] << " ";
    //     // }
    //     if(i && i % 4 == 0)
    //     std::cout << std::endl;
    // }
    delete user;
    delete datar;
    delete datai;
    delete tempr;
    delete tempi;
    return 0;
}
