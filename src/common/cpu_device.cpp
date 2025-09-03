#include "common/cpu_device.h"
#include <iostream>

CpuDevice::CpuDevice(int id) : Device(DeviceType::CPU, id) {
    num_cores = std::thread::hardware_concurrency();
}

void CpuDevice::dev_malloc(void **ptr, size_t size) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
}

void CpuDevice::dev_free(void *ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

void CpuDevice::dev_mem_put_asc(const void *src, void *dst, size_t size, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
}

void CpuDevice::dev_mem_put_asc(const void *src, size_t src_pitch, void *dst, size_t dst_pitch, size_t width_bytes, size_t height, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width_bytes, height, cudaMemcpyDefault, stream));
}

CpuDevice::~CpuDevice() = default;