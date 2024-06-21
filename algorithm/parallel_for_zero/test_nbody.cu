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
#include <omp.h>
#include <cmath>

struct UserData_t : public Basedata_t{
public:
    UserData_t(std::vector<ArrayList*> buf
        ) : buffer(buf){
        }

public:
    std::vector<ArrayList*> buffer;
};

static int length;
double dt;
void cfor_func(Basedata_t* data){
    auto d = (loopData_t*)data;
    auto x1 = ((UserData_t*)(d->buffer))->buffer[0]->get_cdata();
    auto x2 = ((UserData_t*)(d->buffer))->buffer[1]->get_cdata();
    auto x3 = ((UserData_t*)(d->buffer))->buffer[2]->get_cdata();
    auto mass = ((UserData_t*)(d->buffer))->buffer[3]->get_cdata();

    auto v1 = ((UserData_t*)(d->buffer))->buffer[4]->get_cdata();
    auto v2 = ((UserData_t*)(d->buffer))->buffer[5]->get_cdata();
    auto v3 = ((UserData_t*)(d->buffer))->buffer[6]->get_cdata();

    size_t s = d->start;
    size_t e = d->end;
    #pragma omp parallel for simd shared(x1,x2,x3,v1,v2,v3,mass) private(j,i)
    for(int i = s; i < e; ++i){
        double Fx = 0; double Fy = 0; double Fz = 0;
        for(int j = 0; j < length; ++j) {
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
}

__global__ void kernel(size_t s, size_t e, size_t chunk, 
    double* x1, double* x2, double* x3,
    double* v1, double* v2, double* v3,
    double* mass, double dt, double length) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = s + tid * chunk;
    int end = start+chunk < e ? start + chunk : e;

    for(int i = start; i < end; ++i){
        double Fx = 0; double Fy = 0; double Fz = 0;
        for(int j = 0; j < length; ++j) {
            double dx = x1[j] - x1[i];
            double dy = x2[j] - x2[i];
            double dz = x3[j] - x3[i];
            double dst = dx*dx + dy*dy + dz*dz + mass[i];
            double invDist = rsqrt(dst);
            double invDist3 = pow(invDist, 3);
            Fx += dx*invDist3; Fy += dy*invDist3; Fz += dz*invDist3;
        }
        v1[i] += dt * Fx; v2[i] += dt*Fy; v3[i] += dt*Fz;
    }
}

void gfor_func(Basedata_t* data){
    auto d = (loopData_t*)data;
    auto x1 = ((UserData_t*)(d->buffer))->buffer[0]->get_gdata();
    auto x2 = ((UserData_t*)(d->buffer))->buffer[1]->get_gdata();
    auto x3 = ((UserData_t*)(d->buffer))->buffer[2]->get_gdata();
    auto mass = ((UserData_t*)(d->buffer))->buffer[3]->get_gdata();

    auto v1 = ((UserData_t*)(d->buffer))->buffer[4]->get_gdata();
    auto v2 = ((UserData_t*)(d->buffer))->buffer[5]->get_gdata();
    auto v3 = ((UserData_t*)(d->buffer))->buffer[6]->get_gdata();

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
    kernel<<<blocks_required, threads_per_block, 0, stream_>>>(s, e, chunk_size, x1, x2, x3,
        v1, v2, v3, mass, dt, length);
}

int main(int argc, char **argv){
    Framework::init();
    length = std::atoi(argv[1]);
    ArrayList* datax1 = new ArrayList(length);
    ArrayList* datax2 = new ArrayList(length);
    ArrayList* datax3 = new ArrayList(length);

    ArrayList* datav1 = new ArrayList(length);
    ArrayList* datav2 = new ArrayList(length);
    ArrayList* datav3 = new ArrayList(length);
    ArrayList* datamass = new ArrayList(length);
    initialize(datax1, length);
    initialize(datax2, length);
    initialize(datax3, length);
    initialize(datav1, length);
    initialize(datav2, length);
    initialize(datav3, length);
    initialize(datamass, length);
    dt = (double)(rand() % 10);
    UserData_t* user = new UserData_t({datax1, datax2, datax3, datamass,
        datav1, datav2, datav3});
    struct timeval start, end;
    gettimeofday(&start, NULL);
    parallel_for(new loopData_t(0, length, user), cfor_func, gfor_func);
    gettimeofday(&end, NULL);
    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    std::cout << seconds << std::endl;
    delete datax1;
    delete datax2;
    delete datax3;

    delete datav1;
    delete datav2;
    delete datav3;
    delete datamass;
    return 0;
}
