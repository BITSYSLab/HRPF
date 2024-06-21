/*
 * @Author: your name
 * @Date: 2021-11-10 20:25:34
 * @LastEditTime: 2022-02-20 09:55:03
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA\common\cpu_device.cpp
 */
/*
 * @Author: sen
 * @Date: 2020-11-10 20:25:34
 * @LastEditTime: 2021-11-12 16:56:45
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro0/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA\common\cpu_device.cpp
 */
#pragma once

#include "cpu_device.h"

void CpuDevice::dev_malloc(_TYPE** ptr, size_t width, size_t height) {
#if PARALLEL_FOR
    cudaHostAlloc(ptr, width * height * sizeof(_TYPE), cudaHostAllocMapped);
#else
    cudaMallocHost(ptr, width * height * sizeof(_TYPE));
#endif
}

void CpuDevice::dev_malloc(_TYPE** ptr, size_t length) {
#if PARALLEL_FOR
    cudaHostAlloc(ptr, length * sizeof(_TYPE), cudaHostAllocMapped);
#else
    cudaMallocHost(ptr, length * sizeof(_TYPE));
#endif
}

void CpuDevice::dev_free(void *ptr) {
    cudaFreeHost(ptr);
}

void CpuDevice::dev_mem_put(void* dst, size_t dpitch, void* src, size_t spitch,
                size_t width, size_t height) {

    cudaMemcpy2D(dst, dpitch, src, spitch, width, height,
                     cudaMemcpyDeviceToHost);
}

void CpuDevice::dev_mem_put(void* dst, void* src, size_t length) {
    cudaMemcpy(dst, src, sizeof(_TYPE)*length, cudaMemcpyDeviceToHost);
}

void CpuDevice::dev_mem_put_asc(void* dst, size_t dpitch, void* src, size_t spitch,
                size_t width, size_t height) {

    cudaMemcpy2D(dst, dpitch, src, spitch, width, height,
                     cudaMemcpyDeviceToHost);
}

void CpuDevice::dev_mem_put_asc(void* dst, void* src, size_t length) {
    cudaMemcpy(dst, src, sizeof(_TYPE)*length, cudaMemcpyDeviceToHost);
}