/**
 * @file framework.h
 * @author senh (ssh_9711@163.com)
 * @brief  HRPA framework class
 * @version 0.1
 * @date 2022-03-24
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#ifndef _FRAMEWORK
#define _FRAMEWORK

#include <array>
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "framework/task.h"
#include <unordered_map>

/*thread id<--->index*/
extern std::unordered_map<std::thread::id, int> m_map;
// class SplitWSDeque {
// public:
//     int    initialize_size = 1024;
//     int top;
//     int bottom;
//     int split_point;
//     std::atomic_bool all_stolen;
//     bool split_req;
//     std::deque<Task*> activeArray;
// public:
//     SplitWSDeque() {
//         top = 0;
//         bottom = 0;
//         activeArray.resize(initialize_size);
//         split_point = 0;
//         split_req = true;
//         all_stolen = true;
//     }
//     bool cas(int oldV, int newV) {
//         bool flag = __sync_bool_compare_and_swap(&top, oldV, newV);
//         return flag;
//     }

//     Task* steal() {
//         if(this->all_stolen) return nullptr;
//         int t = this->top;
//         int s = this->split_point;
//         if(t < s){
//             if(cas(t, t+1))
//                 return this->activeArray[t];
//             return nullptr;
//         }

//         if(!this->all_stolen) this->all_stolen.store(true);
//         return nullptr;
//     }

//     void grow_shared() {
//         int new_s = (this->split_point + this->bottom) / 2;
//         this->split_point = new_s;
//     }

//     void push(Task* t) {
//         int top = this->top;
//         int bottom = this->bottom;
//         int size = bottom - top;
//         if (size >= activeArray.size()-1){
//             activeArray.resize(2 * size);
//         }

//         activeArray[bottom] = t;
//         this->bottom = bottom + 1;
//         if(all_stolen){
//             grow_shared();
//             this->all_stolen.store(false);
//         }
//     }

//     Task* pop() {
//         int b = this->bottom;
//         b = b - 1;
//         this->bottom = b;
//         int t = this->top;
//         int size = b - t;
//         if (size < 0) {
//             this->bottom = t;
//             return nullptr;
//         }
//         Task* th = activeArray[b];

//         if(this->split_point == b + 1) {
//             if(shrink_shared())
//                 return th;
//         }else
//             return th;
//         this->bottom = t;
//         return th;
//     }

//     bool shrink_shared() {
//         int t = this->top;
//         int s = this->split_point;

//         if(t < s){
//             int new_s = (t + s) / 2;
//             this->split_point = new_s;
//             return true;
//         }
//         this->all_stolen.store(true);
//         return false;
//     }
// };

// class CircularWSDeque {
// public:
//     int    initialize_size = 1024;
//     int top;
//     int bottom;
//     std::deque<Task*> activeArray;
// public:
//     CircularWSDeque() {
//         top = 0;
//         bottom = 0;
//         activeArray.resize(initialize_size);
//     }
//     bool casTop(int oldV, int newV) {
//         bool flag = __sync_bool_compare_and_swap(&top, oldV, newV);
//         return flag;
//     }

//     bool casBottom(int oldV, int newV) {
//         bool flag = __sync_bool_compare_and_swap(&bottom, oldV, newV);
//         return flag;
//     }

//     void push(Task* t) {
//         int top = this->top;
//         int bottom = this->bottom;
//         int size = bottom - top;
//         if (size >= activeArray.size()-1){
//             activeArray.resize(2 * size);
//         }

//         while(true){
//             if(casBottom(bottom, bottom+1)){
//                 activeArray[bottom] = t;
//                 this->bottom = bottom + 1;
//                 break;
//             }
//             bottom = this->bottom;
//         }
//     }

//     Task* steal() {
//         int t = this->top;
//         int b = this->bottom;
//         int size = b - t;
//         if(size <= 0) return nullptr;
//         Task* th = activeArray[t];
//         if(! casTop(t, t+1)){
//             return nullptr;
//         }
//         return th;
//     }

//     Task* pop() {
//         int b = this->bottom;
//         int t = this->top;
//         int size = b - t;
//         if (size < 0) {
//             this->bottom = t;
//             return nullptr;
//         }
//         else if(size == 0) {
//             return nullptr;
//         }

//         if(casBottom(b, b-1)){
//             return activeArray[this->bottom];
//         }

//         return nullptr;
//     }

// };

/**
 * @brief   Maintain information such as \
 *          runtime task queues.
 */
struct helper {
public:
  /*cpu worker num*/
  int m_cpu_threads_num;
  /*gpu worker num*/
  int m_gpu_threads_num;
  /*workers*/
  std::array<std::thread, T_SIZE> m_workers;
  /*shared task queue*/
  std::deque<Problem *> m_task_queue[T_SIZE]; // shared
  /*private task queue*/
  std::deque<Problem *> m_queue_private[T_SIZE];
  /*queue mutex*/
  std::array<std::mutex, T_SIZE> m_task_mutex;
  /*framework worker end flag——
  *每次使用都需初始化_实验中递归算法实例使用的
  *是该终止标志
  */
  bool m_working;
  /*framework worker end flag——一次初始化，
  *多次使用_循环实验使用的是该终止标志*/
  std::atomic_bool terminate;

public:
  /**
   * @brief Construct a new helper object
   *
   */
  helper() {
    terminate = true;//or m_working = true;
    m_cpu_threads_num = c_num;
    m_gpu_threads_num = g_num;
  }

  /**
   * @brief Destroy the helper object
   *
   */
  ~helper() {}

  /**
   * @brief overload = oper
   *
   * @param t : helper obj
   * @return helper&
   */
  helper &operator=(helper &t) {

    terminate = true;
    return t;
  }
};

/**
 * @brief   HRPA framework class
 *
 */
class Framework {
public:
  /**
   * @brief  recursive solve problem
   *          according to interleaving strategy
   * @param problem      :  problem
   * @param interleaving :  parallel strategy
   */
  static void solve(Problem *problem, std::string interleaving);

  /**
   * @brief  per worker exec func
   *
   * @param dev : device string flag
   */
  static void work(std::string dev);

  /**
   * @brief   spawn problem
   *
   * @param problem : problem
   * @param dev     : dev string
   */
  static void spawn(Problem *problem, std::string dev);

  /**
   * @brief  spawn problem
   *
   * @param problem : problem
   * @param id      : thread idx
   */
  static void spawn(Problem *problem, int id);

  /**
   * @brief  wait sub problems finish
   *
   * @param problem : parent problem
   */
  static void wait(Problem *problem);

  /**
   * @brief  cpu worker wait
   *
   * @param problem : parent problem
   */
  static void cpu_wait(Problem *problem);

  /**
   * @brief  gpu worker wait
   *
   * @param problem : parent problem
   */
  static void gpu_wait(Problem *problem);

  /**
   * @brief  framework init
   *
   */
  static void init();

  /**
   * @brief  thread_idx append problems to task queue
   *
   * @param problem : problems
   * @param id      : thread idx
   */
  static void append(std::vector<Problem *> problem, int id);

  /**
   * @brief  framework destroy
   *
   */
  static void destroy();

  // public:
  //     bool terminate;

private:
  Framework(const Framework &);
  Framework(const Framework &&);
  Framework &operator=(const Framework &);
  Framework &operator=(const Framework &&);

private:
  /*parallel strategy*/
  static std::string m_interleaving;
  /*framework config helper*/
  static helper &m_helper;

private:
  /**
   * @brief   solve problem at recursive depth
   *
   * @param _problem : problem
   * @param _depth   : current recursive depth
   */
  static void solve(Problem *_problem, int _depth);

  /**
   * @brief  solve task at recursive depth
   *
   * @param task   : task
   * @param depth  : current recursive depth
   */
  static void solveTask(Task *task, int depth);

  /**
   * @brief  depth >= strategy lens
   *
   * @param _depth  : recursive depth
   * @return true
   * @return false
   */
  static bool shouldRunBaseCase(int _depth);

  /**
   * @brief  delete subproblems
   *
   * @param _subproblems : sub problems
   */
  static void deleteSubproblems(std::vector<Problem *> _subproblems);

  /**
   * @brief Get the Subproblems From Tasks object
   *
   * @param tasks                  : tasks
   * @return std::vector<Problem*> : subproblems
   */
  static std::vector<Problem *>
  getSubproblemsFromTasks(std::vector<Task *> &tasks);

  /**
   * @brief Set the Num Bs object
   *
   * @param _problems  : problems
   * @param _numBs     : num Bs
   */
  static void setNumBs(std::vector<Problem *> _problems, int _numBs);

  // private:
  //     static void gpu_solve(Problem* _problem, int _depth);
};

#define FRAMEWORK_SOLVE void Framework::solve

#define GPU_WAIT void Framework::gpu_wait

#define CPU_WAIT void Framework::cpu_wait

/**
 * @brief    random work steal
 *
 * @param help : global helper obj
 * @param tid  : thread id
 */
void RANDOM_STREAL(helper &help, std::thread::id tid);

/**
 * @brief   work steal strategy
 *
 * @param help   : global helper
 * @param index  : thread idx
 */
void work_steal(helper &help, int index);

/**
 * @brief   help steal strategy
 *
 * @param help   : global helper
 * @param index  : thread idx
 */
void help_steal(helper &help, int index);

#define CLEAR_SUBPROBLEMS(subproblems)                                         \
  int p_size = subproblems.size();                                             \
  for (int j = 0; j < p_size; ++j) {                                           \
    delete subproblems[j];                                                     \
    \ 
        subproblems[j] = nullptr;                                              \
  }                                                                            \
  \        


#define CLEAR_TASKS(tasks) int t_size = tasks.size();                          \
  for (int i = 0; i < t_size; ++i) {                                           \
    if (tasks[i] != nullptr) {                                                 \
      CLEAR_SUBPROBLEMS(tasks[i]->m_problems);                                 \
      delete tasks[i];                                                         \
      tasks[i] = nullptr;                                                      \
    }                                                                          \
    \                  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
  }

/**
 * @brief   record parent and recursive depth
 *
 * @param task   : task
 * @param par    : parent
 * @param depth  : recursive depth
 */
void ADD_PARENT_DEPTH(Task *task, Problem *par, int depth);
#endif
