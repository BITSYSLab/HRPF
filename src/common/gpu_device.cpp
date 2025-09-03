#include "common/gpu_device.h"

GpuDevice::GpuDevice(int id) : Device(DeviceType::GPU, id) {
    initialize_pools();
}

GpuDevice::~GpuDevice() {
    destroy_pools();
}

void GpuDevice::dev_malloc(void **ptr, size_t size) {
    CUDA_CHECK(cudaMalloc(ptr, size));
}

void GpuDevice::dev_free(void *ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

void GpuDevice::dev_mem_put_asc(const void *src, void *dst, size_t size, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
}

void GpuDevice::dev_mem_put_asc(const void *src, size_t src_pitch, void *dst, size_t dst_pitch, size_t width_bytes, size_t height, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width_bytes, height, cudaMemcpyDefault, stream));
}

void GpuDevice::initialize_pools(size_t initial_stream_pool_size, size_t initial_event_pool_size, size_t initial_cublas_handle_pool_size) {
    stream_pool_size_ = initial_stream_pool_size;
    event_pool_size_ = initial_event_pool_size;
    cublas_handle_pool_size_ = initial_cublas_handle_pool_size;

    // 预留空间以避免多次内存重新分配
    stream_pool_.reserve(stream_pool_size_);
    all_managed_streams_.reserve(stream_pool_size_);
    
    for (size_t i = 0; i < stream_pool_size_; ++i) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        stream_pool_.push_back(stream);
        all_managed_streams_.push_back(stream);
    }
    
    event_pool_.reserve(event_pool_size_);
    all_managed_events_.reserve(event_pool_size_);

    for (size_t i = 0; i < event_pool_size_; ++i) {
        cudaEvent_t event;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        event_pool_.push_back(event);
        all_managed_events_.push_back(event);
    }

    cublas_handle_pool_.reserve(cublas_handle_pool_size_);
    all_managed_cublas_handles_.reserve(cublas_handle_pool_size_);

    for (size_t i = 0; i < cublas_handle_pool_size_; ++i) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublas_handle_pool_.push_back(handle);
        all_managed_cublas_handles_.push_back(handle);
    }
}

void GpuDevice::destroy_pools() {
    for (cudaEvent_t event : all_managed_events_) {
        cudaEventDestroy(event);
    }
    all_managed_events_.clear();

    for (cudaStream_t stream : all_managed_streams_) {
        cudaStreamDestroy(stream);
    }
    all_managed_streams_.clear();

    for (cublasHandle_t handle : all_managed_cublas_handles_) {
        cublasDestroy(handle);
    }
    all_managed_cublas_handles_.clear();
}

cudaStream_t GpuDevice::acquire_stream() {
    if (!stream_pool_.empty()) {
        cudaStream_t stream = stream_pool_.back();
        stream_pool_.pop_back();
        return stream;
    }

    // 池为空，动态创建一个
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    all_managed_streams_.push_back(stream); // 追踪它以便销毁
    return stream;
}

void GpuDevice::release_stream(cudaStream_t stream) {
    stream_pool_.push_back(stream);
}

cudaEvent_t GpuDevice::acquire_event() {
    if (!event_pool_.empty()) {
        cudaEvent_t event = event_pool_.back();
        event_pool_.pop_back();
        return event;
    }
    
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    all_managed_events_.push_back(event);
    return event;
}

void GpuDevice::release_event(cudaEvent_t event) {
    event_pool_.push_back(event);
}

cublasHandle_t GpuDevice::acquire_cublas_handle() {
    if (!cublas_handle_pool_.empty()) {
        cublasHandle_t handle = cublas_handle_pool_.back();
        cublas_handle_pool_.pop_back();
        return handle;
    }

    // 池为空，动态创建一个新的
    cublasHandle_t new_handle;
    cublasCreate(&new_handle);
    all_managed_cublas_handles_.push_back(new_handle); // 追踪它以便销毁
    return new_handle;
}

// 将句柄返回到池中
void GpuDevice::realease_cublas_handle(cublasHandle_t handle)
{
    cublasSetStream(handle, nullptr);
    cublas_handle_pool_.push_back(handle);
}