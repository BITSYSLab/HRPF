/*
 * @Author: your name
 * @Date: 2022-03-03 10:09:18
 * @LastEditTime: 2022-03-07 14:25:39
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_12\algorithm\parallel_for\parallel_for_harness.cpp
 */
#pragma once
#include <stdio.h>
#include <sys/time.h>
#include <string>

#include "parallelReduce.cuh"
#include "tool/initializer.h"
// #include "framework/framework.h"

template <typename T>
struct op
{
    __host__ __device__
        T operator()(const T& x, const T& y) const { 
            return x + y;
        }
};

int main(int argc, char **argv){
    
    std::size_t length = std::atoi(argv[1]);
    std::size_t m_size = length * length;
    Framework::init();
    ArrayList* data1 = new ArrayList(m_size);
    initialize(data1, m_size);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    double init_v = 0.0;
    op<double> oper;
    parallel_reduce<op<double>, double>(0, m_size, data1, init_v, oper);
    gettimeofday(&end, NULL);
    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    std::cout << seconds << std::endl;
    delete data1;
    return 0;
}
