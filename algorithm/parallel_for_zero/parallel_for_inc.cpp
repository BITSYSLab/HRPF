/*
 * @Author: your name
 * @Date: 2022-03-02 20:40:18
 * @LastEditTime: 2022-03-03 14:29:32
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_12\algorithm\parallel_for\parallel_for_problem.cpp
 */

#pragma once

#include "parallel_for_inc.h"
#include "framework/framework.h"
#include <iostream>

/**
* for i to n ==> 1-D array SIMD
**/
bool CplusLoopInc::mustRunBaseCase() {
    auto m_d = (loopData_t*)data;
    // std::cout << "task len:" << m_d->end - m_d->start << std::endl;
    return (m_d->end-m_d->start) <= 128;
}

bool CplusLoopInc::canRunBaseCase(int index) {
	return m_mask[index] == 1;
}

std::vector<Problem*> CplusLoopInc::split() {
    auto m_d = (loopData_t*)data;
    auto buffer = m_d->buffer;
    
    std::vector<Problem*> result;
    int iterSize = m_d->end - m_d->start;
    int p = 4;
    int X = 6;
    int q = 4;
    int B = 2*iterSize*(1-q/X) / p / q / (q-1);
    
    int C = iterSize / (X*p);int newStart = m_d->start + C;
    int alloc = C;
    result.emplace_back(new CplusLoopInc(new loopData_t(m_d->start, newStart, buffer), cpu_func, gpu_func,this));
    while(alloc < iterSize) {
        int rest = iterSize - alloc;
        if(rest < C + B){
            result.emplace_back(new CplusLoopInc(new loopData_t(newStart, newStart + rest, buffer), cpu_func, gpu_func,this));
            break;
        }
        C += B;
        result.emplace_back(new CplusLoopInc(new loopData_t(newStart, newStart + C, buffer), cpu_func, gpu_func, this));
        newStart += C;
        B = 2*rest*(1-q/X) / p / q / (q-1);
        alloc += C;
    }

    return result;
}

void CplusLoopInc::merge(std::vector<Problem*>& subproblems){

}

void parallelForI2D(Basedata_t* data, 
    Function cf, Function gf)
{
    // Framework::init();
    auto problem = new CplusLoopInc(data, cf, gf, nullptr);
    // std::cout << "pro" << std::endl;
    Framework::solve(problem, "BBBBBBBBBBBBBBB");
    // std::cout << "solve end" << std::endl;
    Runtime::get_instance().get_gpu()->synchronize();
    delete problem;
}
