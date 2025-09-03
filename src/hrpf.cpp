#include "hrpf.h"
#define SPIN_COUNT 5000

void Framework::init(int mode)
{
    Runtime &runtime = Runtime::get_instance();
    if(mode != 0)   runtime.set_HYBRID_mode();
    // i end to N - 1 because worker N - 1 is the main thread
    for (int i = 0; i < runtime.get_num_workers() - 1; ++i)
    {
        if (i < runtime.get_num_gpu_workers())
        {
            runtime.get_worker(i)->work(Framework::work, i);
        }
        else
        {
            runtime.get_worker(i)->work(Framework::work, i);
        }
    }
}

void Framework::work(const unsigned int worker_id)
{
    Runtime &runtime = Runtime::get_instance();
    Worker *worker = runtime.get_worker(worker_id);
    if (runtime.is_HYBRID()) {
        while (runtime.is_running()) {
            bool found_work = false;
            for (int i = 0; i < SPIN_COUNT; ++i) {
                if (Framework::try_run_one_task(worker)) {
                    found_work = true;
                    break;
                }
            }
            if (found_work) {
                continue;
            }

            uint64_t old_val = runtime.get_task_notifier_value();
            if (!Framework::try_run_one_task(worker)) {
                runtime.wait_for_task(old_val);
            }
        }
    } else {
        while (runtime.is_running()) {
            if(!Framework::try_run_one_task(worker)) {
                std::this_thread::yield();
            }
        }
    }
}

void Framework::submit(Task *task) {
    Runtime &runtime = Runtime::get_instance();
    runtime.dispatch_root(task);
}

void Framework::submit_cpu_only(Task *task) {
    Runtime &runtime = Runtime::get_instance();
    task->set_worker_mode(WorkerMode::CPU_ONLY);
    runtime.get_worker(runtime.get_num_gpu_workers())->push_private_task(task);
}

void Framework::solve(Task *task)
{
    if (task->is_base_case()) {
        task->IO();
        task->run_base_case();
    } else{
        task->split();
        task->dispatch_children();
        Framework::wait_for_children(task);
        task->merge();
    }
    task->complete();
}


void Framework::run_and_wait(Task* task) {
    Runtime &runtime = Runtime::get_instance();
    Worker* self = runtime.get_reserved_worker();
    task->set_worker(self);
    Framework::solve(task);
    Framework::wait(task);
}

void Framework::wait(Task* task) {
        Runtime &runtime = Runtime::get_instance();
    Worker* self = runtime.get_reserved_worker();

    if (runtime.is_HYBRID()) {
        
        while (!task->is_done()) { 
            bool found_other_work = false;
            for(int i = 0; i < SPIN_COUNT; ++i) {
                if(try_run_one_task(self)) {
                    found_other_work = true;
                }
            }

            if (!found_other_work) {
                int current_rc_val = task->load_rc();
                if (current_rc_val != 0) {
                    task->wait_rc(current_rc_val);
                }
            }
        }
    } else {
        while (!task->is_done()) {
            if(!Framework::try_run_one_task(self)) {
                std::this_thread::yield();
            }
        }
    }
    if(task->get_device()->get_type() == DeviceType::GPU) {
        CUDA_CHECK(cudaStreamSynchronize(task->get_stream()));
    }
}

void Framework::wait_for_children(Task* task) {
    Runtime &runtime = Runtime::get_instance();
    Worker* self = task->get_worker();
    if (runtime.is_HYBRID()) {
        while(task->is_children_running()) {
            bool found_other_work = false;
            for(int i = 0; i < SPIN_COUNT; ++i) {
                if(try_run_one_task(self)) {
                    found_other_work = true;
                }
            }

            if(!found_other_work) {
                int current_rc_val = task->load_rc();
                if (current_rc_val > 1) {
                    task->wait_rc(current_rc_val);
                }
            }
        }
    } else {
        while (task->is_children_running()) {
            if(!Framework::try_run_one_task(self)) {
                std::this_thread::yield();
            }
        }
    }
}

void Framework::shutdown()
{
    Runtime &runtime = Runtime::get_instance();
    runtime.terminate();
}

bool Framework::try_run_one_task(Worker *worker)
{
    Task* task = nullptr;
    if (worker->has_private_tasks()) {
        task = worker->pop_private_task();
    } else if (worker->has_shared_tasks()) {
        task = worker->pop_front();
    } else {
        task = worker->steal();
    }
    
    if (task) {
        task->set_worker(worker);
        Framework::solve(task);
        return true;
    }
    return false;
}