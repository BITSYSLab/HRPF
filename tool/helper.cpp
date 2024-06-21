/*
 * @Author: your name
 * @Date: 2022-01-21 20:15:58
 * @LastEditTime: 2022-02-16 14:34:19
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\1num_meachine\HRPA_NEW\tool\helper.cpp
 */
#pragma once

#include "helper.h"
#include <thread>
#include "framework/framework.h"

cublasHandle_t handle(){
// #if ASNC
//     std::thread::id id = std::this_thread::get_id();
//     int index = m_map[id]; 
//     // Runtime& instance = Runtime::get_instance();
                   
//     return Runtime::get_instance().get_gpu()->m_handle[index];
// #else
//     return Runtime::get_instance().get_gpu()->m_handle[0];
// #endif
    return Runtime::get_instance().get_gpu()->m_handle[0];
}

cudaStream_t stream(){
    std::thread::id id = std::this_thread::get_id();
    int index = m_map[id]; 
    // Runtime& instance = Runtime::get_instance();
    // if(index < c_num)               
    return Runtime::get_instance().get_gpu()->m_curr_stream[index];
    // return Runtime::get_instance().get_gpu()->m_handle_;
}
