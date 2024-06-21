/*
 * @Author: your name
 * @Date: 2022-03-02 20:40:18
 * @LastEditTime: 2022-03-03 14:29:32
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_12\algorithm\parallel_for\parallel_for_problem.cpp
 */

#pragma once

#include "parallel_for_zb.h"
#include "framework/framework.h"
#include <iostream>

/**
* for i to n ==> 1-D array SIMD
**/
bool CplusLoop::mustRunBaseCase() {
    auto m_d = (loopData_t*)data;
    // std::cout << "task len:" << m_d->end - m_d->start << std::endl;
    return (m_d->end-m_d->start) <= 128;
}

bool CplusLoop::canRunBaseCase(int index) {
	return m_mask[index] == 1;
}

std::vector<Problem*> CplusLoop::split() {
    auto m_d = (loopData_t*)data;
    auto buffer = m_d->buffer;
    
    size_t mid = (m_d->start + m_d->end) / 2;
    // std::cout << "mid:" << mid << std::endl;
    std::vector<Problem*> result(2);
    result[0] = new CplusLoop(new loopData_t(m_d->start, mid, buffer), cpu_func, gpu_func,this);
    result[1] = new CplusLoop(new loopData_t(mid, m_d->end, buffer),cpu_func, gpu_func ,this);
    return result;
}

void CplusLoop::merge(std::vector<Problem*>& subproblems){

}

void parallel_for(Basedata_t* data, 
    Function cf, Function gf)
{
    // Framework::init();
    auto problem = new CplusLoop(data, cf, gf, nullptr);
    // std::cout << "pro" << std::endl;
    Framework::solve(problem, "BBBBBBBBBBBBBBB");
    // std::cout << "solve end" << std::endl;
    Runtime::get_instance().get_gpu()->synchronize();
    delete problem;
}
