/*
 * @Author: sen
 * @Date: 2021-11-06 15:21:52
 * @LastEditTime: 2021-11-23 20:45:49
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA\common\runtime.cpp
 */
#pragma once

#include "runtime.h"
#include <iostream>

Runtime::Runtime() {
    srand(time(0));
    runtime_state_ = RuntimeState::STOPED;
    
    cpu_ = new CpuDevice();
    gpu_ = new GpuDevice();
}

Runtime::~Runtime() {
    delete cpu_;
    delete gpu_;
    cpu_ = nullptr;
    gpu_ = nullptr;
    // std::cout << "runtime delete..." << std::endl;
}

CpuDevice* Runtime::get_cpu() {
    return cpu_;
}

GpuDevice* Runtime::get_gpu() {
    return gpu_;
}

