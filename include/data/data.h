#pragma once

#include "common/device.h"
#include "common/enum.h"
#include <vector>
#include <memory>
#include <functional>

// Forward declaration to avoid circular dependency
class Runtime;

template <typename T>
class BaseData
{
public:
    using DataPair_t = std::pair<T *, MemState>;

    virtual ~BaseData();

    // --- Data Access ---
    T *get_cdata() { return cpu_pair_.first; }
    const T *get_cdata() const { return cpu_pair_.first; }
    T *get_gdata() { return gpu_pair_.first; }
    const T *get_gdata() const { return gpu_pair_.first; }

    const DataPair_t& get_cpu_pair() const { return cpu_pair_; }
    const DataPair_t& get_gpu_pair() const { return gpu_pair_; }
    
    BaseData<T> *get_parent() { return parent_; }
    const BaseData<T> *get_parent() const { return parent_; }

    BaseData<T> *get_child(size_t index)
    {
        return (index < children_.size()) ? children_[index].get() : nullptr;
    }
    const BaseData<T> *get_child(size_t index) const
    {
        return (index < children_.size()) ? children_[index].get() : nullptr;
    }

    void access(Device *d, MemAccess ma, cudaStream_t stream) {
        // CPU需要使用同步的copy
        bool has_children = !children_.empty();
        if(has_children) {
            // 如果有子数据，这里就指示子数据执行真正的数据copy
            for (auto& child : children_) {
                child->access(d, ma, stream);
            }
        }
        DeviceType type = d->get_type();
        DataPair_t &target_pair = (type == DeviceType::CPU) ? cpu_pair_ : gpu_pair_;
        DataPair_t &other_pair = (type == DeviceType::CPU) ? gpu_pair_ : cpu_pair_;
        // 只有没有子数据的data才可以copy
        switch (ma) {
            case MemAccess::W:
                target_pair.second = MemState::EXCLUSIVE;
                other_pair.second = MemState::INVALID;
                break; 
            case MemAccess::R:
                if (other_pair.second == MemState::EXCLUSIVE)
                {
                    if(!has_children) copy_from_asc(other_pair.first, target_pair.first, d, stream);
                    target_pair.second = MemState::SHARED;
                    other_pair.second = MemState::SHARED;
                }
                break;
            case MemAccess::RW:
                if (other_pair.second == MemState::EXCLUSIVE)
                {
                    if(!has_children) copy_from_asc(other_pair.first, target_pair.first, d, stream);
                }
                target_pair.second = MemState::EXCLUSIVE;
                other_pair.second = MemState::INVALID;
                break;
        }
    }

protected:
    /* only for inherit */
    explicit BaseData(size_t total_bytes);

    BaseData(T* data, size_t total_bytes);

    BaseData(T* cpu_data, T* gpu_data, size_t total_bytes, DeviceType type);

    BaseData(BaseData<T> *parent, T *cpu_ptr, T *gpu_ptr)
        : parent_(parent), mallocd_(false)
    {
        cpu_pair_.first = cpu_ptr;
        gpu_pair_.first = gpu_ptr;
        cpu_pair_.second = parent->cpu_pair_.second;
        gpu_pair_.second = parent->gpu_pair_.second;
    }

    // both copy funtions use stream, but copy_from will synchronize
    // so cpu workers use copy_from to wait until data is ready and then work
    // gpu workers can work after memcpy because stream makes it sequntially
    virtual void copy_from(T *src, T *dst, Device *d, cudaStream_t stream) = 0;
    virtual void copy_from_asc(T *src, T *dst, Device *d, cudaStream_t stream) = 0;

    DataPair_t cpu_pair_;
    DataPair_t gpu_pair_;
    bool mallocd_;
    BaseData<T> *parent_;
    std::vector<std::unique_ptr<BaseData<T>>> children_;

    // 禁止拷贝和移动，因为这个类管理着原始指针和所有权
    BaseData(const BaseData &) = delete;
    BaseData &operator=(const BaseData &) = delete;
    BaseData(BaseData &&) = delete;
    BaseData &operator=(BaseData &&) = delete;
};