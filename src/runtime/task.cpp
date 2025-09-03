#include "runtime/task.h"
#include "runtime/worker.h"
#include "runtime/runtime.h"
#include <stdexcept>

// 根任务构造函数
Task::Task(std::string strategy, std::vector<BaseData<_TYPE>*> data, std::vector<MemAccess> modes, OffloadPolicy policy)
    : parent_(nullptr), worker_(nullptr), depth_(0), 
      global_strategy_(std::move(strategy)), policy_(policy), sibling_id_(0),
      data_(std::move(data)), access_modes_(std::move(modes)) {
    if (data_.size() != access_modes_.size()) {
        throw std::invalid_argument("Data and access modes vectors must have the same size.");
    }
    if (!global_strategy_.empty()) {
        local_strategy_ = global_strategy_[0] == 'B' ? TaskStrategy::BFS : TaskStrategy::DFS;
    }
}

// 子任务构造函数
Task::Task(Task *parent, std::vector<BaseData<_TYPE>*> data, std::vector<MemAccess> modes, size_t sibling_id) 
    : parent_(parent), worker_(nullptr), global_strategy_(parent->global_strategy_),
      depth_(parent->depth_ + 1), policy_(parent_->policy_), sibling_id_(2*parent_->sibling_id_ + sibling_id),
      data_(std::move(data)), access_modes_(std::move(modes)) {
    if (data_.size() != access_modes_.size()) {
        throw std::invalid_argument("Data and access modes vectors must have the same size.");
    }
    if (depth_ < global_strategy_.size()) {
        local_strategy_ = global_strategy_[depth_] == 'B' ? TaskStrategy::BFS : TaskStrategy::DFS;
    } else {
        local_strategy_ = parent->local_strategy_;
    }
    parent->rc_.fetch_add(1, std::memory_order_relaxed);
}

void Task::IO() {
    Device* dev = get_device();
    cudaStream_t stream = dev->get_type() == DeviceType::GPU ? stream_ : cudaStreamPerThread;
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i]->access(dev, access_modes_[i], stream);
    }
    if(dev->get_type() == DeviceType::CPU) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

void Task::set_worker(Worker* worker) {
    worker_ = worker;
    Device* dev = worker->get_device();
    if (dev->get_type() == DeviceType::GPU) {
        GpuDevice* gpu_dev = static_cast<GpuDevice*>(dev);
        // 如果是默认流，则分配专门的流；如果不是，则说明该task被手动设置流了，不要更改
        if(stream_ == cudaStreamPerThread) stream_ = gpu_dev->acquire_stream();
        split_done_event_ = gpu_dev->acquire_event();
        merge_done_event_ = gpu_dev->acquire_event();
    } else {
        stream_ = cudaStreamPerThread;
        split_done_event_ = nullptr;
        merge_done_event_ = nullptr;
    }
}

// 是否可以卸载到cpu，副作用：如果可以则还会把该任务准备好卸载
inline bool Task::should_offload() {
    if( policy_.high_water_mark > 1 && sibling_id_ != 0 &&sibling_id_ % policy_.high_water_mark == 0 ) {
        depth_ -= policy_.additional_depth;
        return true;
    }
    // TODO 阈值判断
    return false;
}

void Task::dispatch_children() {
    if (children_.empty()) return;
    Runtime &runtime = Runtime::get_instance();
    if (is_bfs()) {
        
        if (worker_->get_device()->get_type() == DeviceType::CPU) {
            worker_->push_front(children_[0].get());
            for (size_t i = 1; i < children_.size(); ++i) {
                runtime.get_worker(runtime.get_next_worker_id())->push_back(children_[i].get());
            }
        } else {
            for(const auto& child: children_) {
                if (child->is_base_case() && child->should_offload()) {
                    // get_num_gpu_workers获得的就是第一个cpu worker的ID
                    runtime.get_worker(runtime.get_num_gpu_workers())->push_back(child.get());
                } else {
                    worker_->push_front(child.get());
                }
            }
        }
        
    } else {
        // DFS策略不会触发卸载
        std::vector<Task*> child_tasks;
        child_tasks.reserve(children_.size());
        for (const auto& child : children_) {
            child_tasks.push_back(child.get());
        }
        worker_->push_front_batch(child_tasks);
    }
    runtime.notify_all_workers();
}

void Task::complete() {  
    if (get_device()->get_type() == DeviceType::GPU && parent_) {
        CUDA_CHECK(cudaEventRecord(merge_done_event_, stream_));
        CUDA_CHECK(cudaStreamWaitEvent(parent_->get_stream(), merge_done_event_));
    }
    if (rc_.fetch_sub(1, std::memory_order_release) == 1) {
        rc_.notify_all();
    }

    if (parent_) {
        if (parent_->rc_.fetch_sub(1, std::memory_order_release) == 2) {
            parent_->rc_.notify_all();
        }
    }
    // TODO bug fix
    if(get_device()->get_type() == DeviceType::GPU) {
        GpuDevice* gpu_dev = static_cast<GpuDevice*>(get_device());
        gpu_dev->release_stream(stream_);
        gpu_dev->release_event(split_done_event_);
        gpu_dev->release_event(merge_done_event_);
    }
}

Device* Task::get_device() const {
    if (!worker_) return Runtime::get_instance().get_cpu();
    return worker_->get_device();
}

void Task::run() {
    IO();
    run_base_case();
    complete();
}