#pragma once

#include "common/device.h"
#include <memory_resource>
#include <cublas_v2.h>

class GpuDevice final : public Device {
public:
    explicit GpuDevice(int id);
    ~GpuDevice() override;

    void dev_malloc(void **ptr, size_t size) override;
    void dev_free(void *ptr) override;
    
    void dev_mem_put_asc(const void *src, void *dst, size_t size, cudaStream_t stream) override;
    void dev_mem_put_asc(const void *src, size_t src_pitch, void *dst, size_t dst_pitch, size_t width_bytes, size_t height, cudaStream_t stream) override;

    
    cudaStream_t acquire_stream();
    void release_stream(cudaStream_t stream);

    cudaEvent_t acquire_event();
    void release_event(cudaEvent_t event);

    cublasHandle_t acquire_cublas_handle();
    void realease_cublas_handle(cublasHandle_t handle);

private:
    void initialize_pools(size_t initial_stream_pool_size = 64, size_t initial_event_pool_size = 128, size_t initial_cublas_handle_pool_size = 32);
    void destroy_pools();

    size_t stream_pool_size_;
    size_t event_pool_size_;
    size_t cublas_handle_pool_size_;

    // 使用std::vector作为池。后进先出（LIFO）
    std::vector<cudaStream_t> stream_pool_;
    std::vector<cudaEvent_t> event_pool_;
    std::vector<cublasHandle_t> cublas_handle_pool_;

    // 仍然需要追踪所有创建的资源，以便在析构时正确销毁
    std::vector<cudaStream_t> all_managed_streams_;
    std::vector<cudaEvent_t> all_managed_events_;
    std::vector<cublasHandle_t> all_managed_cublas_handles_;
};