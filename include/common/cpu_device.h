#pragma once

#include "common/device.h"
#include <thread>
#include <cstddef>
#include <cstring>

class CpuDevice final : public Device {
public:
    explicit CpuDevice(int id);
    ~CpuDevice() override;

    void dev_malloc(void **ptr, size_t size) override;
    void dev_free(void *ptr) override;
    
    void dev_mem_put_asc(const void *src, void *dst, size_t size, cudaStream_t stream) override;
    void dev_mem_put_asc(const void *src, size_t src_pitch, void *dst, size_t dst_pitch, size_t width_bytes, size_t height, cudaStream_t stream) override;

    [[nodiscard]] int get_num_cores() const { return num_cores; }

private:
    int num_cores;
};