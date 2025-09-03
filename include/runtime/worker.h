#pragma once

#include <thread>
#include <iostream>
#include <stack>
#include <random>
#include "common/device.h"
#include "common/shared_task_queue.h"
#include "runtime/runtime.h"

class Task;

class Worker {
public:
    Worker(unsigned int id, Device* device);

    ~Worker();

    [[nodiscard]] unsigned int id() const { return id_; }
    [[nodiscard]] Device* get_device() const { return device_; }

    // only gpu worker should use private task queue
    [[nodiscard]] bool has_private_tasks() const { return !private_task_queue_.empty(); }
    [[nodiscard]] Task* pop_private_task();
    void push_private_task(Task* task) { private_task_queue_.push(task); }

    // all workers can use shared task queue
    [[nodiscard]] bool has_shared_tasks() const { return !shared_task_queue_.empty() ; }
    // for work-first
    [[nodiscard]] Task* pop_front() { return shared_task_queue_.pop_front(); }
    // for dfs distribution
    bool push_front(Task* task) { 
        if (device_->get_type() == DeviceType::GPU) {
            private_task_queue_.push(task);
            return true;
        }
        return shared_task_queue_.push_front(task); 
    }
    bool push_front_batch(std::vector<Task*> tasks) { 
        if (device_->get_type() == DeviceType::GPU) {
            for (auto task : tasks) private_task_queue_.push(task);
            return true;
        }
        return shared_task_queue_.push_front_batch(tasks.begin(), tasks.end()); 
    }
    // for help-first
    [[nodiscard]] Task* pop_back() { return shared_task_queue_.pop_back(); }
    // for bfs distribution
    bool push_back(Task* task) { return shared_task_queue_.push_back(task); }
    

    template<typename Callable, typename... Args>
    void work(Callable&& func, Args&&... args) {
        thread_ = std::thread(std::forward<Callable>(func), std::forward<Args>(args)...);
    }

    [[nodiscard]] Task* steal();

    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
    Worker(Worker&&) = delete;
    Worker& operator=(Worker&&) = delete;

private:
    unsigned int id_;
    Device* device_;
    std::thread thread_;
    SharedTaskQueue<Task*> shared_task_queue_;
    std::stack<Task*> private_task_queue_;
};