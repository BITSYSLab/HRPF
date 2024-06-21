/**
 * @file problem.h
 * @author senh (ssh_9711@163.com)
 * @brief  problem class declare
 * @version 0.1
 * @date 2022-03-23
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include "common/data.h"
#include "common/device.h"
#include "common/enum.h"
#include "tool/helper.h"

//#include "task.h"
#include <atomic>
#include <functional>
#include <vector>

//#include "runtime.h"
#include "config.h"
#include <bitset>
// typedef void (*func_t)();
class Task;

/**
 * @brief the base data type defination,
 *          which used to define
 */
struct Basedata_t {
  virtual ~Basedata_t() {}
};

/**
 * @brief common function(cpu & gpu) interface
 *
 */
typedef typename std::function<void(Basedata_t *)> Function;

/**
 * @brief problem interface declare
 *
 */
class Problem {
public:
  // typedef typename std::function<void(Basedata_t*)> Function;
  /**
   * @brief Destroy the Problem object
   *
   */
  virtual ~Problem() {}

  /**
   * @brief per problem base case imple
   *
   */
  virtual void runBaseCase();

  /**
   * @brief subproblems merge result to parent problem.
   *
   * @param subproblems : sub problems
   */
  virtual void merge(std::vector<Problem *> &subproblems) = 0;

  /**
   * @brief used to release memory space(default -- null func)
   *
   * @param subproblems
   */
  virtual void mergePostPro(std::vector<Problem *> subproblems) {}

  /**
   * @brief problem split -> subproblems
   *
   * @return std::vector<Problem*> : sub problems
   */
  virtual std::vector<Problem *> split() = 0;

  /**
   * @brief seq split
   *      similiar to split
   * @return std::vector<Problem *>
   */
  virtual std::vector<Problem *> splitSequential();

  /**
   * @brief seq merge
   *
   * @param subproblems
   */
  virtual void mergeSequential(std::vector<Problem *> subproblems);

  /**
   * @brief hard recursive end condition
   *
   * @return true
   * @return false
   */
  virtual bool mustRunBaseCase();

  /**
   * @brief is or not run base case \
   *              --> relate to worker
   * @param index : thread idx
   * @return true
   * @return false
   */
  virtual bool canRunBaseCase(int index);

  /**
   * @brief record problem run in which device
   *
   * @param dev : device
   */
  virtual void record_device(Device *dev);

  /**
   * @brief record device according to dev char
   *
   * @param dev : device char
   */
  virtual void record_device(char dev);

  /**
   * @brief prepare IO data
   *
   */
  virtual void prepare_data();

  /**
   * @brief exec the problem
   *
   */
  virtual void exec();

  /**
   * @brief Set the depth of recursion
   *
   * @param _depth
   */
  virtual void set_depth(int _depth);

  /**
   * @brief exec the problem
   *
   * @param m_data : IO data
   * @param cfunc  : cpu func
   * @param gfunc  : gpu func
   */
  virtual void run_task(Basedata_t *m_data, Function cfunc, Function gfunc);

  /**
   * @brief exec the problem
   *
   */
  virtual void run();

  /**
   * @brief exec the problem
   *
   * @param c : device char
   */
  virtual void run(char c);

  /**
   * @brief async exec the problem
   *
   * @param c : device char
   */
  virtual void runAsc(char c);

  /**
   * @brief
   *      Reduce cpu func
   */
  virtual void CPU_FUNC() {}

  /**
   * @brief
   *      Reduce gpu func
   */
  virtual void GPU_FUNC() {}

public:
  /**
   * @brief record per input data -> call input()
   *
   */
  virtual void Input() {} // ?告知哪些是输入数据

  /**
   * @brief record per output data -> call output()
   *
   */
  virtual void Output() {} // ?告知哪些是输出数据

  /**
   * @brief I/O data
   *
   * @param m_data
   */
  virtual void IO(Basedata_t *m_data) {}

  /**
   * @brief in data
   *
   * @tparam data_type : datastructure type
   * @param ha         : data
   */
  template <class data_type> void input(data_type *&ha) {
    ha->access(this->device, MemAccess::R);
  }

  /**
   * @brief out data
   *
   * @tparam data_type : datastructure type
   * @param ha         : data
   */
  template <class data_type> void output(data_type *&ha) {
    ha->access(this->device, MemAccess::W);
  }

  /**
   * @brief async in data
   *
   * @tparam data_type : datastructure type
   * @param ha         : data
   */
  template <class data_type> void inputAsc(data_type *&ha) {
    ha->accessAsc(this->device, MemAccess::R);
  }

  /**
   * @brief async out data
   *
   * @tparam data_type : datastructure type
   * @param ha         : data
   */
  template <class data_type> void outputAsc(data_type *&ha) {
    ha->accessAsc(this->device, MemAccess::W);
  }

  /**
   * @brief record parent problem
   *
   * @param it : parent
   */
  void addParent(Problem *it) { parent = it; }

public:
  /*sub problems*/
  std::vector<Problem *> childs;
  /*problem data*/
  Basedata_t *data;
  /*run device*/
  Device *device = nullptr;

public:
  /**
   * @brief append child
   *
   * @param th : child problem
   */
  void add_child(Problem *th);

  /**
   * @brief
   *      not used
   * @return int
   */
  int notify();

  /**
   * @brief
   *      not used
   */
  void finish();

  /**
   * @brief
   *      not used
   */
  void notify_childs();

  /**
   * @brief   add cpu & gpu func
   *
   * @param cf : cpu_func
   * @param gf : gpu_func
   */
  void add_operation(Function cf, Function gf);

public:
  /*recursion depth*/
  int depth{0}; // the current step
  /*recursive or not*/
  bool flag{true}; // need next recursion?

  /*cpu imple*/
  Function cpu_func;
  /*gpu imple*/
  Function gpu_func;
  /*worker---recursive flag*/
  std::bitset<T_SIZE> m_mask;
  /*parent problem*/
  Problem *parent{nullptr};
  /*not used, replaced with rc*/
  std::atomic<int> deps = ATOMIC_VAR_INIT(0);
  /*finish flag*/
  std::atomic<bool> done = ATOMIC_VAR_INIT(false);
  /*deps*/
  std::atomic<int> rc = ATOMIC_VAR_INIT(0); // num of the subproblems
};

/**
 * @brief
 *      deprecate
 */
struct P_data {
public:
  Problem *m_problem;
  int t_idx;
  P_data(Problem *problem, int index) : m_problem(problem), t_idx(index) {}
};

/*deprecate*/
void runDevice(P_data m_data);