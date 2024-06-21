/**
 * @file task.h
 * @author senh (ssh_9711@163.com)
 * @brief  task class declare
 * @version 0.1
 * @date 2022-03-23
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include "framework/problem.h"
#include <initializer_list>
#include <vector>

#include "common/runtime.h"
#include "config.h"
#include <thread>

/**
 * @brief
 *     A task object is a set of problems, running in sequence.
 * @version 0.1
 * @author senh (ssh_9711@163.com)
 * @date 2022-03-23
 * @copyright Copyright (c) 2022
 */
class Task {
public:
  /*set of problems*/
  std::vector<Problem *> m_problems;
  /*recursive flag*/
  bool flag; // is recursive?
  /*task size*/
  size_t m_size;

public:
  /**
   * @brief Construct a new Task object
   *
   * @param set: set of problems
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  Task(std::initializer_list<Problem *> set) : m_problems(set), flag(true) {
    m_size = m_problems.size();
    for (int i = 0; i < m_size; ++i) {
      m_problems[i]->flag = flag;
    }
  }

  /**
   * @brief Destroy the Task object
   *
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  ~Task() {
    // int size = m_problems.size();
    // for(int i = 0; i < size; ++i){
    //     if(m_problems[i]){
    //         delete m_problems[i];
    //         m_problems[i] = nullptr;
    //     }
    // }
  }

  /**
   * @brief Construct a new Task object
   *
   * @param problem : set of problems
   * @param f : recursive flag
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  Task(Problem *problem, bool f = true) : flag(f), m_size(1) {
    m_problems.resize(1);
    m_problems[0] = problem;
    m_problems[0]->flag = flag;
  }

  /**
   * @brief Get the problems object
   *
   * @return std::vector<Problem*>
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  std::vector<Problem *> get_problems() { return m_problems; }

  /**
   * @brief Set the recursive flag object
   *
   * @param f : bool flag
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  void set_flag(bool f) { flag = f; }

  /**
   * @brief run a task
   *
   * @param par : parent problem
   * @param c  : device flag
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  void run(Problem *par, char c);
};
