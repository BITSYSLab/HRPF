#pragma once

#include "common/enum.h"
#include "common/device.h"
#include "common/cuda_utils.h"
#include "offload_policy.h"
#include "data/data.h"
#include <atomic>
#include <memory>
#include <vector>
#include <string>

class Worker;
class Runtime;

class Task {
public:
    Task(std::string strategy, 
         std::vector<BaseData<_TYPE>*> data, 
         std::vector<MemAccess> modes,
         OffloadPolicy policy = OffloadPolicy());

    Task(Task* parent, 
         std::vector<BaseData<_TYPE>*> data, 
         std::vector<MemAccess> modes,
         size_t sibling_id);

    virtual ~Task() = default;
   
    // --- core logic ---
    virtual void split() = 0;
    virtual void merge() = 0;
    virtual void run_base_case() = 0;

    [[nodiscard]] virtual bool is_base_case() const {
        return depth_ >= global_strategy_.size();
    }
    
    virtual void IO();

    void dispatch_children();

    void complete();

    void run();
    
    int load_rc() {  return rc_.load(std::memory_order_acquire); }

    void wait_rc(int old_val) { rc_.wait(old_val, std::memory_order_acquire); }

    [[nodiscard]] inline bool should_offload();

    // --- Getter and Setters ---
    [[nodiscard]] bool is_done() const { return rc_.load(std::memory_order_acquire) == 0; }

    [[nodiscard]] bool is_bfs() const { return local_strategy_ == TaskStrategy::BFS; }

    [[nodiscard]] bool is_root() const { return depth_ == 0; }

    [[nodiscard]] bool is_children_running() const { return rc_.load(std::memory_order_acquire) > 1; }

    [[nodiscard]] int get_depth() const { return depth_; }

    [[nodiscard]] const std::string& get_global_strategy() const { return global_strategy_; }

    [[nodiscard]] Worker* get_worker() const { return worker_; }

    void set_worker(Worker* worker);

    [[nodiscard]] const std::vector<BaseData<_TYPE>*>& get_data() const { return data_; }

    [[nodiscard]] const std::vector<MemAccess>& get_access_modes() const { return access_modes_; }

    [[nodiscard]] Task* get_parent() const { return parent_; }

    [[nodiscard]] Device* get_device() const;

    void set_stream(cudaStream_t stream) { stream_ = stream; }

    [[nodiscard]] cudaStream_t get_stream() const { return stream_; }

    [[nodiscard]] cudaEvent_t get_split_done_event() const { return split_done_event_; }

    [[nodiscard]] cudaEvent_t get_merge_done_event() const { return merge_done_event_; } 

    [[nodiscard]] bool is_hetero() const { return worker_mode_ == WorkerMode::HETERO; }

    [[nodiscard]] bool is_cpu_only() const { return worker_mode_ == WorkerMode::CPU_ONLY; }

    [[nodiscard]] bool is_gpu_only() const { return worker_mode_ == WorkerMode::GPU_ONLY; }

    void set_worker_mode(WorkerMode mode) { worker_mode_ = mode; }

protected:
    // 完成这个task所需要用到的数据以及访问他们的模式
    std::vector<BaseData<_TYPE>*> data_;
    std::vector<MemAccess> access_modes_;

    Task* parent_;
    std::vector<std::unique_ptr<Task>> children_;

    // assigned after dispatch
    Worker* worker_ = nullptr;

    // for distribution and scheduling
    int depth_ = 0;
    std::string global_strategy_;
    TaskStrategy local_strategy_ = TaskStrategy::BFS;

    // for synchronization and overlap
    cudaStream_t stream_ = cudaStreamPerThread;
    cudaEvent_t split_done_event_;  // 可以进行子任务/运行base case
    cudaEvent_t merge_done_event_;  // 可以进行下一步merge或者根任务完成
    /* 
        rc_ = 0: 该任务已经完成
        rc_ = 1: 默认情况或者所有子任务已经完成
        rc_ > 1: 该任务有子任务正在运行
    */
    std::atomic<int> rc_ = ATOMIC_VAR_INIT(1);

    WorkerMode worker_mode_ = WorkerMode::HETERO;

    // for offload from gpu to cpu
    size_t sibling_id_; // 表示该Task在该层的ID，在最深层用于判断是否应当split
    OffloadPolicy policy_;
};