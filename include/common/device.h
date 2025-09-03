#pragma once

#include "enum.h"
#include "cuda_utils.h"
#include <stddef.h>

using DeviceID = int;

class Device {
public:
    explicit Device(DeviceType type, int id) : device_type_(type), device_id_(id) {}
    virtual ~Device() = default;

    [[nodiscard]] virtual DeviceID get_id() const { return device_id_; };
    [[nodiscard]] virtual DeviceType get_type() const { return device_type_; };

    // --- malloc and free ---
    virtual void dev_malloc(void **ptr, size_t size) = 0;
    virtual void dev_free(void *ptr) = 0;

    // --- Memory Copy (Asynchronous) ---
    virtual void dev_mem_put_asc(const void *src, void *dst, size_t size, cudaStream_t stream) = 0;
    virtual void dev_mem_put_asc(const void *src, size_t src_pitch, void *dst, size_t dst_pitch, size_t width_bytes, size_t height, cudaStream_t stream) = 0;

protected:
    Device() = default;
    // copy and move is not allowed
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;

    /* 唯一的ID */
    DeviceID device_id_;
    /* 设备类型, CPU/GPU */
    DeviceType device_type_;
};