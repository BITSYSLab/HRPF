#pragma once

#include "runtime/runtime.h"
#include "data/arraylist.h"
#include "data/matrix.h"

class Framework{
public:
  /* Init the framework, includeing setup workers and work mode*/
  static void init(int mode = 0);

  /* The working loop of the workers */
  static void work(const unsigned int worker_id);

  /* Run the task. It will block until task is finished. */
  static void run_and_wait(Task* task);

  /* submit a task, non-blocking, default on gpu first*/
  static void submit(Task* task);

  // submit a task on cpu
  static void submit_cpu_only(Task* task);

  /* Solve a task */
  static void solve(Task* task);

  /* wait the task to be **finished**; wait while working */
  static void wait(Task* task);

  /* Wait all children of the task finished; wait while working */
  static void wait_for_children(Task* task);


  static void parallel_for(Task* task);

  // static void parallel_for_I(std::vector<LoopData*> data);

  // static void parallel_for_D(std::vector<LoopData*> data);

  /* Shutdown the framework */
  static void shutdown();

private:
  static bool try_run_one_task(Worker* worker);

};
