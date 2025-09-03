#include "data/data.h"
#include "runtime/runtime.h"

// Template implementation needs to be in header file for templates
template <typename T>
BaseData<T>::~BaseData()
{
    children_.clear();

    if (mallocd_)
    {
        auto &runtime = Runtime::get_instance();
        auto cpu = runtime.get_cpu();
        auto gpu = runtime.get_gpu();
        if (cpu_pair_.first) cpu->dev_free(cpu_pair_.first);
        if (gpu_pair_.first) gpu->dev_free(gpu_pair_.first);
    }
}

template <typename T>
BaseData<T>::BaseData(size_t total_bytes) : parent_(nullptr), mallocd_(true)
{
    auto &runtime = Runtime::get_instance();
    auto cpu = runtime.get_cpu();
    auto gpu = runtime.get_gpu();
    
    cpu->dev_malloc(reinterpret_cast<void**>(&(cpu_pair_.first)), total_bytes);
    gpu->dev_malloc(reinterpret_cast<void**>(&(gpu_pair_.first)), total_bytes);
    
    cpu_pair_.second = MemState::INVALID;
    gpu_pair_.second = MemState::INVALID;
}

template <typename T>
BaseData<T>::BaseData(T* data, size_t total_bytes) : parent_(nullptr), mallocd_(false)
{
    auto &runtime = Runtime::get_instance();
    auto cpu = runtime.get_cpu();
    auto gpu = runtime.get_gpu();
    cpu_pair_.first = data;
    gpu->dev_malloc(reinterpret_cast<void**>(&(gpu_pair_.first)), total_bytes);
    cpu_pair_.second = MemState::EXCLUSIVE;
    gpu_pair_.second = MemState::INVALID;
}

template <typename T>
BaseData<T>::BaseData(T* cpu_data, T* gpu_data, size_t total_bytes, DeviceType type) : parent_(nullptr), mallocd_(false)
{
    cpu_pair_.first = cpu_data;
    gpu_pair_.first = gpu_data;
    if(type == DeviceType::CPU){
        cpu_pair_.second = MemState::EXCLUSIVE;
        gpu_pair_.second = MemState::INVALID;
    } else {
        cpu_pair_.second = MemState::INVALID;
        gpu_pair_.second = MemState::EXCLUSIVE;
    }
}

template class BaseData<double>;