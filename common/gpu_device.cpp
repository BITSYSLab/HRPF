/*
 * @Author: your name
 * @Date: 2021-11-11 16:06:40
 * @LastEditTime: 2022-02-20 13:47:32
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA\common\gpu_device.cpp
 */
#pragma once

#include "gpu_device.h"
#include "framework/problem.h"
#include "framework/task.h"
#include "framework/framework.h"
#include <thread>

void GpuDevice::dev_malloc(_TYPE** ptr, size_t width, size_t height) {
    cudaMalloc(ptr, width * height * sizeof(_TYPE)); 
}

void GpuDevice::dev_malloc(_TYPE** ptr, size_t length) {
    cudaMalloc(ptr, length * sizeof(_TYPE));
}

void GpuDevice::dev_free(void *ptr) {
    cudaFree(ptr);
}

void GpuDevice::synchronize() {
    cudaDeviceSynchronize();
}

void GpuDevice::recycle_stream(cudaStream_t stream) {
    {
        std::lock_guard<std::mutex> lk(m_streams_mutex);
        assert(m_idle_streams.size() < STREAM_NUM_);
        m_idle_streams.push_back(stream);
    }
    m_streams_cv.notify_all();
}

void CUDART_CB call_back(cudaStream_t stream, cudaError_t status, void* userData) {
    GpuDevice* gpu = ((CallBackData*)userData)->gpu;
    Problem* p = ((CallBackData*)userData)->task;
    gpu->recycle_stream(stream);
	if(p->parent) p->parent->rc.fetch_sub(1);
    //gpu->recycle_stream(stream);
    p->done = true;
    delete (CallBackData*)userData;
}

void CUDART_CB call_back_single(cudaStream_t stream, cudaError_t status, void* userData) {
    GpuDevice* gpu = ((CallBackData*)userData)->gpu;
    Problem* p = ((CallBackData*)userData)->task;
    gpu->recycle_stream(stream);
    
	if(p->parent) p->parent->rc.fetch_sub(1);
    //gpu->recycle_stream(stream);
    //p->done = true;
    delete p;
    delete (CallBackData*)userData;
}

void CUDART_CB call_back_mer(cudaStream_t stream, cudaError_t status, void* userData) {
    GpuDevice* gpu = ((CallBackData*)userData)->gpu;
    Problem* p = ((CallBackData*)userData)->task;
    Task* t = ((CallBackData*)userData)->t_task;
    gpu->recycle_stream(stream);
	p->rc.fetch_sub(((CallBackData*)userData)->m_size);
     //########## new add #########//
    if(t) {
        for(int i = 0; i < t->m_size; ++i){
            delete t->m_problems[i];
            t->m_problems[i] = nullptr;
        }
        delete t;
    }
    //gpu->recycle_stream(stream);
    delete (CallBackData*)userData;
}

void GpuDevice::dev_mem_put(void* dst, size_t dpitch, void* src, size_t spitch,
                size_t width, size_t height) {
    cudaMemcpy2D(dst, dpitch, src, spitch, width, height,
                          cudaMemcpyHostToDevice);
}

void GpuDevice::dev_mem_put(void* dst, void* src, size_t length) {
    cudaMemcpy(dst, src, sizeof(_TYPE)*length, cudaMemcpyHostToDevice);
}

void GpuDevice::dev_mem_put_asc(void* dst, size_t dpitch, void* src, size_t spitch,
                size_t width, size_t height) {
    
    std::thread::id id = std::this_thread::get_id();
    int t_idx = m_map[id];
    cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                          cudaMemcpyHostToDevice, m_curr_stream[t_idx]);
}

void GpuDevice::dev_mem_put_asc(void* dst, void* src, size_t length) {
    std::thread::id id = std::this_thread::get_id();
    int t_idx = m_map[id];
    cudaMemcpyAsync(dst, src, sizeof(_TYPE)*length, cudaMemcpyHostToDevice, m_curr_stream[t_idx]);
}
