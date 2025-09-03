#include "runtime/runtime.h"
#include <thread>

Runtime& Runtime::get_instance() {
    static Runtime instance;
    if(instance.initialized_) {
        return instance;
    }
    instance.initialize();
    return instance;
}

void Runtime::initialize() {
    if (initialized_) return;

    terminate_.store(false);

    cpu_device_ = std::make_unique<CpuDevice>(0);
    gpu_device_ = std::make_unique<GpuDevice>(1);
    num_workers_ = std::thread::hardware_concurrency();
    num_gpu_workers_ = 1;
    num_cpu_workers_ = num_workers_ - num_gpu_workers_;
    for (unsigned int i = 0; i < num_workers_; i++)
    {
        if(i < num_gpu_workers_) {
            workers_.emplace_back(std::make_unique<Worker>(i, gpu_device_.get()));
        } else {
            workers_.emplace_back(std::make_unique<Worker>(i, cpu_device_.get()));
        }
    }
    workers_.shrink_to_fit();

    initialized_ = true;
}

void Runtime::shutdown() {
    if (!initialized_) return;

    terminate_.store(true);
}

void Runtime::dispatch_root(Task* root_task) {
    if (!initialized_) {
        initialize();
    }

    if (!workers_.empty()) {
        int worker_id = 0;
        root_task->set_worker(workers_[worker_id].get());
        workers_[0]->push_private_task(root_task);
    }
    notify_all_workers();
}

unsigned int Runtime::get_next_worker_id() {
    unsigned int idx = next_worker_idx_.fetch_add(1, std::memory_order_relaxed);
    return (idx % num_cpu_workers_) + num_gpu_workers_;
}

void Runtime::notify_all_workers() {
    task_notifier_.fetch_add(1, std::memory_order_release);
    task_notifier_.notify_all();
}

void Runtime::wait_for_task(uint64_t old_val) {
    task_notifier_.wait(old_val, std::memory_order_acquire);
}

uint64_t Runtime::get_task_notifier_value() const {
    return task_notifier_.load(std::memory_order_acquire);
}