/**
 * @file framework.cpp
 * @author senh (ssh_9711@163.com)
 * @brief   framework class imple
 * @version 0.1
 * @date 2022-03-24
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include "framework.h"

/*****log: thread::id:index*****/
std::unordered_map<std::thread::id, int> m_map;

std::string Framework::m_interleaving;
helper help_global;
helper &Framework::m_helper = help_global;

bool Framework::shouldRunBaseCase(int _depth) {
  return _depth >= Framework::m_interleaving.size();
}

void Framework::init() {
  auto init_thread = [&](helper &help) {
    //help.m_working = true;
    help.terminate = true;
    int size = help.m_workers.size();
    int c_size = help.m_cpu_threads_num;
    for (int i = 0; i < c_size; ++i) {
      std::string dev = "CPU";
      help.m_workers[i] = std::thread(&Framework::work, dev);
      m_map[help.m_workers[i].get_id()] = i;
      help.m_workers[i].detach();
    }

    for (int i = c_size; i < size; ++i) {
      help.m_workers[i] = std::thread(&Framework::work, "GPU");
      m_map[help.m_workers[i].get_id()] = i;
      help.m_workers[i].detach();
    }
  };

  init_thread(help_global);
}

/*only call when use terminate stop all threads*/
void Framework::destroy() {
  // for(int i = 0; i < T_SIZE; ++i){
  //     for(int k = 0; k < help_global.m_queue_delete[i].size(); ++k) {
  //         CLEAR_SUBPROBLEMS(help_global.m_queue_delete[i][k]);
  //     }
  // }
  Framework::m_helper.terminate.store(false);
}

void Framework::spawn(Problem *problem, int id) {
  std::unique_lock<std::mutex> lk(Framework::m_helper.m_task_mutex[id]);
  Framework::m_helper.m_task_queue[id].push_front(problem);
  lk.unlock();
}

void Framework::append(std::vector<Problem *> problem, int id) {
  std::unique_lock<std::mutex> lk(Framework::m_helper.m_task_mutex[id]);
  Framework::m_helper.m_task_queue[id].insert(
      Framework::m_helper.m_task_queue[id].begin(), problem.begin(),
      problem.end());
  lk.unlock();
}

void Framework::solveTask(Task *task, int depth) {
  std::vector<Problem *> problems = task->get_problems();
  for (auto &p : problems) {
    solve(p, depth);
  }
}

std::vector<Problem *>
Framework::getSubproblemsFromTasks(std::vector<Task *> &task) {
  std::vector<Problem *> subproblems;
  for (auto &t : task) {
    subproblems.insert(subproblems.end(), t->m_problems.begin(),
                       t->m_problems.end());
    delete t;
    t = nullptr;
  }
  return subproblems;
}

void Framework::wait(Problem *problem) {
  auto cur_id = std::this_thread::get_id();
  const int cpu_index = m_map[cur_id];

  // int c_size = Framework::m_helper.m_cpu_threads_num;
  int c_size = c_num;
  if (cpu_index >= c_size) {
    Framework::gpu_wait(problem);
  } else {
    Framework::cpu_wait(problem);
  }
}

void Framework::work(const std::string dev) {

  if (dev == "GPU") {
    /* code */
    auto gpu = Runtime::get_instance().get_gpu();
    std::thread::id id = std::this_thread::get_id();
    int index = m_map[id]; // Framework::m_helper.m_working
    while (Framework::m_helper.terminate) {
      if (!Framework::m_helper.m_queue_private[index].empty()) {
        Problem *task = Framework::m_helper.m_queue_private[index].back();
        Framework::m_helper.m_queue_private[index].pop_back();
        task->record_device(gpu);
        Framework::solve(task, task->depth);
      } else {
        /*************lock************/
        std::unique_lock<std::mutex> ul(
            Framework::m_helper.m_task_mutex[index]);
        if (!Framework::m_helper.m_task_queue[index].empty()) {
          Problem *task = Framework::m_helper.m_task_queue[index].back();
          Framework::m_helper.m_task_queue[index].pop_back();
          ul.unlock();

          task->record_device(gpu);
          Framework::solve(task, task->depth);
        } else {
          ul.unlock();
          help_steal(help_global, index);
        }
      }
    }
  } else {
    auto cpu = Runtime::get_instance().get_cpu();
    // std::cout << "cpu get" << std::endl;
    std::thread::id id = std::this_thread::get_id();
    int index = m_map[id];
    // std::cout << "xpu index" << index <<
    // std::endl;Framework::m_helper.m_working
    while (Framework::m_helper.terminate) {
      /*******************lock******************/
      // std::cout << "s:" << Framework::m_helper.m_queue_private[index].size()
      // << std::endl;
      if (!Framework::m_helper.m_queue_private[index].empty()) {
        Problem *task = Framework::m_helper.m_queue_private[index].front();
        Framework::m_helper.m_queue_private[index].pop_front();
        task->record_device(cpu);
        // std::cout << "task->de:" << task->depth << std::endl;
        Framework::solve(task, task->depth);
        // std::cout << "cpu solve" << std::endl;
      } else {
        std::unique_lock<std::mutex> ul(
            Framework::m_helper.m_task_mutex[index]);
        if (!Framework::m_helper.m_task_queue[index].empty()) {
          Problem *task = Framework::m_helper.m_task_queue[index].front();
          Framework::m_helper.m_task_queue[index].pop_front();
          ul.unlock();
          task->record_device(cpu);
          Framework::solve(task, task->depth);
        } else {
          ul.unlock();
          work_steal(help_global, index);
        }
      }
    }
  }
}

void Framework::solve(Problem *_problem, int _depth) {
  std::thread::id id = std::this_thread::get_id();
  int t_idx = m_map[id];
  /*log: end condition*/
  // std::cout << "enter solve " << t_idx << std::endl;
  if (_problem->canRunBaseCase(t_idx) ||
      _depth >= Framework::m_interleaving.size() ||
      _problem->mustRunBaseCase()) {
    // std::cout << "rub bace:" << t_idx << std::endl;
    _problem->runBaseCase();
    return;
  }

  std::vector<Problem *> subproblems;
  if (Framework::m_interleaving[_depth] == 'B') {
    _problem->rc = 0;
    // std::cout << "B:" << t_idx << std::endl;
    auto tasks = _problem->split();
#if ASNC
    while (_problem->rc) {
      // std::cout << "_p_rc:" << _problem->rc << std::endl;
    };
#endif
    int size = tasks.size();
    _problem->rc = tasks.size();
    // std::cout << "split:" << size << " idx:" << t_idx << std::endl;
    for (int i = 0; i < size; ++i) {
      tasks[i]->depth = _depth + 1;
      int dst = i % T_SIZE;
      if (dst < c_num && !help_global.m_task_queue[dst].empty())
        Framework::spawn(tasks[i], c_num);
      else
        Framework::spawn(tasks[i], dst);
    }
    // if(size <= g_num){
    //     for(int i = 0; i < size; ++i){
    //         int dst = (i % g_num) + c_num;
    //         Framework::spawn(tasks[i], dst);
    //     }
    // } else {
    //     int i = 0;
    //     for(; i < g_num; ++i){
    //         int dst = (i % g_num) + c_num;
    //         Framework::spawn(tasks[i], dst);
    //     }

    //     for(; i < size; ++i){
    //         int dst = i % c_num;
    //         if(!help_global.m_task_queue[dst].empty())
    //             Framework::spawn(tasks[i], c_num);
    //         else
    //             Framework::spawn(tasks[i], dst);
    //     }
    // }

    Framework::wait(_problem);
    // std::cout << "wait end:" << t_idx << std::endl;
    _problem->merge(tasks);
    // std::cout << "merge end..." << t_idx << std::endl;

#if ASNC
    while (_problem->rc) {
      // std::cout << "_p_rc:" << _problem->rc << std::endl;
    }
#endif
    _problem->mergePostPro(tasks);
    CLEAR_SUBPROBLEMS(tasks);
    _problem->done = true;
    if (_problem->parent)
      _problem->parent->rc.fetch_sub(1);
  } else {
    _problem->rc = 0;
    auto tasks = _problem->split();
    int size = tasks.size();
#if ASNC
    while (_problem->rc)
      ;
#endif
    _problem->rc = size;
    for (int i = 0; i < size; ++i) {
      // tasks[i]->device = _problem->device;
      tasks[i]->depth = _depth + 1;
      // Framework::solve(tasks[i], _depth+1);
      // ADD_PARENT_DEPTH(tasks[i], _problem, _depth + 1);
    }
    Framework::append(tasks, t_idx);
    Framework::wait(_problem);
#if ASNC
    while (_problem->rc)
      ;
#endif
    _problem->merge(tasks);

#if ASNC
    while (_problem->rc)
      ;
#endif
    _problem->mergePostPro(tasks);
    CLEAR_SUBPROBLEMS(tasks);
    _problem->done = true;
    if (_problem->parent)
      _problem->parent->rc.fetch_sub(1);
  }
}

FRAMEWORK_SOLVE(Problem *problem, std::string interleaving) {
  Framework::m_interleaving = interleaving;
  /**lock**/
  // std::unique_lock<std::mutex> lk(Framework::m_helper.m_task_mutex[0]);
  // Framework::m_helper.m_task_queue[0].push_front(problem);
  // lk.unlock();
  // std::cout << "push" << std::endl;
  Framework::m_helper.m_queue_private[0].push_front(problem);
  // std::cout << "push end" << std::endl;
  /*log: exit*/
  while (!(problem->done)) {
    Framework::m_helper.terminate.store(true);//Framework::m_helper.m_working = true;
  }
  //Framework::m_helper.m_working = false;//only use m_working stop all threads
}

CPU_WAIT(Problem *problem) {
  std::atomic_int &rc = problem->rc;
  auto cpu = Runtime::get_instance().get_cpu();
  std::thread::id id = std::this_thread::get_id();
  helper &help = Framework::m_helper;
  int index = m_map[id];
  while (rc > 0) {
    /***************lock***************/
    // std::cout << "wait index:" << index << " rc:" <<rc << std::endl;
    Problem *task = nullptr;
    // if(!help.m_queue_private[index].empty()) {
    //     task = help.m_queue_private[index].front();
    //     help.m_queue_private[index].pop_front();
    //     task->record_device(cpu);
    //     Framework::solve(task, task->depth);
    // } else {
    std::unique_lock<std::mutex> ul(help.m_task_mutex[index]);
    if (!help.m_task_queue[index].empty()) {
      task = help.m_task_queue[index].front();
      help.m_task_queue[index].pop_front();
      ul.unlock();

      task->record_device(cpu);
      Framework::solve(task, task->depth);

    } else {
      ul.unlock();
      work_steal(help, index);
    }
    // }
  }
}

GPU_WAIT(Problem *problem) {
  std::atomic_int &rc = problem->rc;
  auto gpu = Runtime::get_instance().get_gpu();
  std::thread::id id = std::this_thread::get_id();
  helper &help = Framework::m_helper;
  int index = m_map[id];
  while (rc > 0) {
    /***************lock***************/
    Problem *task = nullptr;
    if (!help.m_queue_private[index].empty()) {
      task = help.m_queue_private[index].back();
      help.m_queue_private[index].pop_back();
      task->record_device(gpu);
      Framework::solve(task, task->depth);
    } else {
      std::unique_lock<std::mutex> ul(help.m_task_mutex[index]);
      if (!help.m_task_queue[index].empty()) {
        task = help.m_task_queue[index].back();
        help.m_task_queue[index].pop_back();
        ul.unlock();

        task->record_device(gpu);
        Framework::solve(task, task->depth);

      } else {
        ul.unlock();
        help_steal(help, index);
      }
    }
  }
}

void RANDOM_STREAL(helper &help, std::thread::id tid) {
  srand((unsigned)time(NULL));
  // int src = rand() % T_SIZE;
  int src;
  int dst = m_map[tid];
  // if(dst < c_num)
  //     src = rand() % c_num;
  // else{
  //     src = rand() % T_SIZE;
  // }
  src = rand() % c_num;
  if (src == dst)
    return;

  Problem *th = nullptr;
  std::unique_lock<std::mutex> ul(help.m_task_mutex[src]);
  if (!help.m_task_queue[src].empty()) {
    th = help.m_task_queue[src].back();
    help.m_task_queue[src].pop_back();
    ul.unlock();

    {
      std::lock_guard<std::mutex> lk(help.m_task_mutex[dst]);
      help.m_task_queue[dst].push_front(th);
    }
  } else {
    ul.unlock();
  }
}

void work_steal(helper &help, int index) {
  srand((unsigned)time(NULL));
  int src = rand() % T_SIZE;

  Problem *th = nullptr;
  std::unique_lock<std::mutex> ul(help.m_task_mutex[src]);
  if (!help.m_task_queue[src].empty()) {
    th = help.m_task_queue[src].front();
    help.m_task_queue[src].pop_front();
    ul.unlock();

    {
      std::lock_guard<std::mutex> lk(help.m_task_mutex[index]);
      help.m_task_queue[index].push_front(th);
    }
  } else {
    ul.unlock();
  }
}

void help_steal(helper &help, int index) {
  srand((unsigned)time(NULL));
  int src = -1;
  if (c_num == 0)
    src = rand() % T_SIZE;
  else
    src = rand() % c_num;

  Problem *th = nullptr;
  std::unique_lock<std::mutex> ul(help.m_task_mutex[src]);
  if (!help.m_task_queue[src].empty()) {
    th = help.m_task_queue[src].back();
    help.m_task_queue[src].pop_back();
    ul.unlock();

    help.m_queue_private[index].push_front(th);
  } else {
    ul.unlock();
  }
}

void ADD_PARENT_DEPTH(Task *task, Problem *par, int depth) {
  int size = task->m_problems.size();
  par->rc.fetch_add(size);
  for (int l = 0; l < size; ++l) {
    // task->m_problems[l]->device = par->device;
    task->m_problems[l]->depth = depth;
  }
}
