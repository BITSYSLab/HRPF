/*
 * @Author: your name
 * @Date: 2021-11-22 10:59:05
 * @LastEditTime: 2022-02-24 10:20:20
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_NEW\framework\problem.cpp
 */
#pragma once

#include "problem.h"
#include "framework.h"
#include "task.h"

bool Problem::mustRunBaseCase() {
    return false;
}

bool Problem::canRunBaseCase(int index) {
    return true;
}

void Problem::mergeSequential(std::vector<Problem*> subproblems) {
    merge(subproblems);
}

std::vector<Problem*> Problem::splitSequential() {
    // std::vector<Problem*> ret;
    std::vector<Problem*> subproblems;
    // std::vector<Task*> tasks = split();
    // for(auto &t : tasks) {
    //     subproblems.insert(subproblems.end(), t->m_problems.begin(), t->m_problems.end());
    //     delete t;
    //     t = nullptr;
    // }
        
    // for(auto &subp : subproblems)
    return subproblems;
}

void Problem::record_device(Device* dev) {
    device = dev;
}

void Problem::record_device(char dev) {
    switch (dev)
    {
    case 'g':
        /* code */
        device = Runtime::get_instance().get_gpu();
        break;
    
    case 'c':
        /* code */
        device = Runtime::get_instance().get_cpu();
        break;

    default:
        break;
    }
}

void Problem::prepare_data() {
    Input();
    Output();
}

void Problem::exec() {
    if(device->get_type() == DeviceType::CPU) {
        cpu_func(data);
    } else {
        gpu_func(data);
    }
}

void Problem::set_depth(int _depth) {
    depth = _depth;
}

void Problem::finish() {
    notify_childs();
}

void Problem::notify_childs() {
    for(auto& child : childs) {
        child->notify();
    }
}

int Problem::notify() {
    deps.fetch_sub(1);
    return deps;
}

void Problem::add_operation(Function cf, Function gf) {
    cpu_func = cf;
    gpu_func = gf;
}

void Problem::run() {
    IO(data);
    if(device->get_type() == DeviceType::CPU) {
        cpu_func(data);
    } else {
        gpu_func(data);
    }
}

void Problem::run(char c) {
    if(c == 'c') {
        device = Runtime::get_instance().get_cpu();
        IO(data);
        cpu_func(data);
    } else {
        device = Runtime::get_instance().get_gpu();
        IO(data);
        gpu_func(data);
    }
}

void Problem::runAsc(char c) {
    if(c == 'c') {
        device = Runtime::get_instance().get_cpu();
        IO(data);
        cpu_func(data);
    } else {
        auto gpu = Runtime::get_instance().get_gpu();
        device = gpu;
        std::thread::id id = std::this_thread::get_id();
        int t_idx = m_map[id];
        
        while(true)
        {
            std::unique_lock<std::mutex> lk(gpu->m_streams_mutex);
            {
                if(!gpu->m_idle_streams.empty()) {
                    gpu->m_curr_stream[t_idx] = gpu->m_idle_streams.back();
                    gpu->m_idle_streams.pop_back();
                    break;
                } 
                else{
                    gpu->m_streams_cv.wait(lk, [=](){return  gpu->m_idle_streams.size() != 0;});
                }
            }
        }
        parent->rc.fetch_add(1);
        
        IO(data);
        if(HANDLE_NUM_) cublasSetStream(gpu->m_handle[0], gpu->m_curr_stream[t_idx]);
        gpu_func(data);
        CallBackData *cbd = new CallBackData(gpu, this, 1);
        cudaStreamAddCallback(gpu->m_curr_stream[t_idx], call_back_single, cbd, 0);
    }
}

void runDevice(P_data m_data) {
    auto problem = m_data.m_problem;
    int t_idx = m_data.t_idx;
#if ASNC
    auto gpu = Runtime::get_instance().get_gpu();
    while(true)
    {
        std::unique_lock<std::mutex> lk(gpu->m_streams_mutex);
        {
            if(!gpu->m_idle_streams.empty()) {
                gpu->m_curr_stream[t_idx] = gpu->m_idle_streams.back();
                gpu->m_idle_streams.pop_back();
                break;
            } 
            else{
                gpu->m_streams_cv.wait(lk, [=](){return  gpu->m_idle_streams.size() != 0;});
            }
        }
    }
	problem->IO(problem->data);
    if(HANDLE_NUM_) cublasSetStream(gpu->m_handle[0], gpu->m_curr_stream[t_idx]);
    problem->gpu_func(problem->data);
    CallBackData *cbd = new CallBackData(gpu, problem, 0);
    cudaStreamAddCallback(gpu->m_curr_stream[t_idx], call_back, cbd, 0);
    // cudaStreamSynchronize(gpu->m_curr_stream[t_idx]);
#else
	problem->IO(problem->data);
    problem->gpu_func(problem->data);
    problem->done = true;
    if(problem->parent) problem->parent->rc.fetch_sub(1);
#endif
}

void Problem::runBaseCase() {
    if(device == nullptr && parent)
        device = parent->device;
    
    //IO(data);
    if(device->get_type() == DeviceType::CPU)
    {
        //exec();
        IO(data);
#if _REDUCE
        CPU_FUNC();
#else
        // std::cout << "exec cpu" << std::endl;
		cpu_func(data);
#endif
        done = true;
        if(parent) parent->rc.fetch_sub(1);
        return;
    }

#if ASNC
    std::thread::id id = std::this_thread::get_id();
    int t_idx = m_map[id];
    auto gpu = (GpuDevice*)device;
    // {
    //     std::unique_lock<std::mutex> lk(gpu->m_streams_mutex);
    //     gpu->m_streams_cv.wait(lk, [=](){return  gpu->m_idle_streams.size() != 0;});  // �ȴ��п���������
    // }
    // bool stream_flag = false;
    while(true)
    {
        std::unique_lock<std::mutex> lk(gpu->m_streams_mutex);
        {
            if(!gpu->m_idle_streams.empty()) {
                gpu->m_curr_stream[t_idx] = gpu->m_idle_streams.back();
                gpu->m_idle_streams.pop_back();
                break;
            } 
            else{
                gpu->m_streams_cv.wait(lk, [=](){return  gpu->m_idle_streams.size() != 0;});
            }
        }
    }
	IO(data);
    if(HANDLE_NUM_) cublasSetStream(gpu->m_handle[0], gpu->m_curr_stream[t_idx]);
#if _REDUCE
        GPU_FUNC();
#else
        // std::cout << "exec gpu" << std::endl;
		gpu_func(data);
#endif
    CallBackData *cbd = new CallBackData(gpu, this, 0);
    cudaStreamAddCallback(gpu->m_curr_stream[t_idx], call_back, cbd, 0);
    // cudaStreamSynchronize(gpu->m_curr_stream[t_idx]);
#else
	IO(data);
#if _REDUCE
        GPU_FUNC();
#else
		gpu_func(data);
#endif
    done = true;
    if(parent) parent->rc.fetch_sub(1);
#endif
}

//not use this func
void Problem::run_task(Basedata_t* m_data, Function cfunc, Function gfunc) {
    this->IO(m_data);
    if(device->get_type() == DeviceType::CPU) {
        cfunc(m_data);
        return;
    }
    gfunc(m_data);
    delete m_data;
}
