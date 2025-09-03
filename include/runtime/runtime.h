#pragma once

#include "common/cpu_device.h"
#include "common/gpu_device.h"
#include "runtime/worker.h"
#include "runtime/task.h"
#include <vector>
#include <memory>
#include <atomic>

class Runtime {
public:
    static Runtime& get_instance();

    void initialize();

    void shutdown();

    void dispatch_root(Task* root_task);

    [[nodiscard]] CpuDevice* get_cpu() const { return cpu_device_.get(); }
    [[nodiscard]] GpuDevice* get_gpu() const { return gpu_device_.get(); }

    [[nodiscard]] int get_num_workers() const { return num_workers_; }
    [[nodiscard]] const std::vector<std::unique_ptr<Worker>>& get_workers() const { return workers_; }
    [[nodiscard]] Worker* get_worker(unsigned int id) const {
        if (id < workers_.size()) {
            return workers_[id].get();
        }
        return nullptr;
    }
    [[nodiscard]] Worker* get_reserved_worker() const {
        if (!workers_.empty()) {
            return workers_.back().get(); // The last worker is reserved for the main thread
        }
        return nullptr;
    }

    [[nodiscard]] bool is_initialized() const { return initialized_; }
    [[nodiscard]] bool is_running() const { return !terminate_.load(); }
    void terminate() { terminate_.store(true); notify_all_workers(); }

    [[nodiscard]] int get_num_cpu_workers() const { return num_cpu_workers_; }
    [[nodiscard]] int get_num_gpu_workers() const { return num_gpu_workers_; }

    [[nodiscard]] unsigned int get_next_worker_id();

        // for notification
    void notify_all_workers();
    void wait_for_task(uint64_t old_val);
    [[nodiscard]] uint64_t get_task_notifier_value() const;

    void set_HYBRID_mode() { mode_ = WorkMode::HYBRID; }

    [[nodiscard]] bool is_HYBRID()  { return mode_ == WorkMode::HYBRID; }


private:
    Runtime() = default;
    ~Runtime() = default;

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    
    // Devices
    std::unique_ptr<CpuDevice> cpu_device_;
    std::unique_ptr<GpuDevice> gpu_device_;

    // Workers
    // The last worker is reserved for the main thread
    unsigned int num_workers_;
    std::vector<std::unique_ptr<Worker>> workers_;
    unsigned int num_cpu_workers_;
    unsigned int num_gpu_workers_;

    // state flags
    bool initialized_ = false;
    std::atomic<bool> terminate_ = true;    // only written by runtime itself, read by any thread

    alignas(64) std::atomic<unsigned int> next_worker_idx_{0};
    std::atomic<uint64_t> task_notifier_{0};

    WorkMode mode_ = WorkMode::ACTIVE;
};