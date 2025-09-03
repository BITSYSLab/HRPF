#include "runtime/worker.h"

Worker::Worker(unsigned int id, Device* device): id_(id), device_(device) {}

Worker::~Worker() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

Task* Worker::pop_private_task() {
    if(private_task_queue_.empty()) return nullptr;
    Task* task = private_task_queue_.top();
    private_task_queue_.pop();
    return task;
}

Task* Worker::steal() {
    Runtime &runtime = Runtime::get_instance();
    const int num_workers = runtime.get_num_workers();
    const int num_gpu_workers = runtime.get_num_gpu_workers();
    Task* stolen_task = nullptr;

    thread_local std::mt19937 generator(std::random_device{}() + id_);
    std::uniform_int_distribution<unsigned int> distribution(num_gpu_workers, num_workers - 1);
    unsigned int victim_id = distribution(generator);

    if (victim_id != id_ && victim_id >= num_gpu_workers) {
        Worker* victim = runtime.get_worker(victim_id);
        if (victim) {
            if (get_device()->get_type() == DeviceType::GPU) {
                // help-first TODO
                // stolen_task = victim->pop_back();
            } else{
                // work-first
                stolen_task = victim->pop_front();
            }
        }
    }
    return stolen_task;
}