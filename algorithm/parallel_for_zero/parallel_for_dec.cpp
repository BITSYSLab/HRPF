/*
 * @Author: your name
 * @Date: 2022-03-02 20:40:18
 * @LastEditTime: 2022-03-03 14:29:32
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_12\algorithm\parallel_for\parallel_for_problem.cpp
 */

#pragma once

#include "parallel_for_dec.h"
#include "framework/framework.h"
#include <iostream>

/**
* for i to n ==> 1-D array SIMD
**/
bool CplusLoopDec::mustRunBaseCase() {
    auto m_d = (loopData_t*)data;
    // std::cout << "task len:" << m_d->end - m_d->start << std::endl;
    return (m_d->end-m_d->start) <= 128;
}

bool CplusLoopDec::canRunBaseCase(int index) {
	return m_mask[index] == 1;
}

std::vector<Problem*> CplusLoopDec::split() {
    auto m_d = (loopData_t*)data;
    auto buffer = m_d->buffer;
    
    std::vector<Problem*> result;
    int iterSize = m_d->end - m_d->start;
    int p = 4;
    int F = iterSize / (2*p);
    int L = 1;
    int N = 2 * iterSize / (F + L);
    int D = (F - L) / (N - 1);
    int newStart = m_d->start + F;
    int C = F;
    int alloc = F;
    result.emplace_back(new CplusLoopDec(new loopData_t(m_d->start, newStart, buffer), cpu_func, gpu_func,this));
    while(alloc < iterSize) {
        int rest = iterSize - alloc;
        if(rest < C - D){
            result.emplace_back(new CplusLoopDec(new loopData_t(newStart, newStart + rest, buffer), cpu_func, gpu_func,this));
            break;
        }
        C -= D;
        result.emplace_back(new CplusLoopDec(new loopData_t(newStart, newStart + C, buffer), cpu_func, gpu_func, this));
        newStart += C;
        F = rest / (2*p);
        N = 2 * rest / (F + L);
        D = (F - L) / (N - 1);
        alloc += C;
    }

    return result;
}

void CplusLoopDec::merge(std::vector<Problem*>& subproblems){

}

void parallelForR2D(Basedata_t* data, 
    Function cf, Function gf)
{
    // Framework::init();
    auto problem = new CplusLoopDec(data, cf, gf, nullptr);
    // std::cout << "pro" << std::endl;
    Framework::solve(problem, "BBBBBBBBBBBBBBB");
    // std::cout << "solve end" << std::endl;
    Runtime::get_instance().get_gpu()->synchronize();
    delete problem;
}
