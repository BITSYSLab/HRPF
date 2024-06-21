/**
 * @file runtime.h
 * @author senh (ssh_9711@163.com)
 * @brief  runtime system management class.
 * @version 0.1
 * @date 2022-03-23
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

// #include <condition_variable>
#include "cpu_device.h"
#include "device.h"
#include "enum.h"
#include "gpu_device.h"
#include <cuda_runtime.h>
#include <string>

/**
 * @brief
 *      Runtime system class.
 * @version 0.1
 * @author senh (ssh_9711@163.com)
 * @date 2022-03-23
 * @copyright Copyright (c) 2022
 */
class Runtime final {
public:
  /**
   * @brief Get the instance object
   *
   * @return Runtime& : runtime instance ref
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  static Runtime &get_instance() {
    static Runtime instance;
    return instance;
  }

  /**
   * @brief Destroy the Runtime object
   *
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  ~Runtime();

  /**
   * @brief Get the cpu object
   *
   * @return CpuDevice*
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  CpuDevice *get_cpu();

  /**
   * @brief Get the gpu object
   *
   * @return GpuDevice*
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  GpuDevice *get_gpu();

private:
  /*runtime system state*/
  RuntimeState runtime_state_;
  /*cpu device object*/
  CpuDevice *cpu_;
  /*gpu device object*/
  GpuDevice *gpu_;
  /*not used*/
  size_t ntasks_;
  // std::mutex tasks_mutex_;
  // std::condition_variable wait_device_;

private:
  Runtime();
  Runtime(const Runtime &);
  Runtime(const Runtime &&);
  Runtime &operator=(const Runtime &);
  Runtime &operator=(const Runtime &&);
};
