/**
 * @file device.h
 * @author senh (ssh_9711@163.com)
 * @brief   device interface declare.
 * @version 0.1
 * @date 2022-03-23
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once
#include "enum.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
 * @brief
 *      device class declare
 * @version 0.1
 * @author senh (ssh_9711@163.com)
 * @date 2022-03-23
 * @copyright Copyright (c) 2022
 */
class Device {
protected:
  /*device type*/
  DeviceType device_type;
  /*device state*/
  DeviceState device_state;

public:
  /**
   * @brief Construct a new Device object
   *
   * @param dt : enum device type
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  Device(DeviceType dt) : device_type(dt){};

  /**
   * @brief Destroy the Device object
   *
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  virtual ~Device() {}

  /**
   * @brief Get the device type
   *
   * @return DeviceType
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  DeviceType get_type() const { return device_type; }
  // virtual void dev_malloc() = 0;
  // virtual void dev_free(void *) = 0;

  /**
   * @brief matrix data asnc copy oper
   *
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  virtual void dev_mem_put_asc(void *, size_t, void *, size_t, size_t, size_t) {
  }

  /**
   * @brief arraylist data asnc copy oper
   *
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  virtual void dev_mem_put_asc(void *, void *, size_t) {}

  /**
   * @brief matrix data sync copy oper
   *
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  virtual void dev_mem_put(void *, size_t, void *, size_t, size_t, size_t) = 0;

  /**
   * @brief arraylist data sync copy oper
   *
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  virtual void dev_mem_put(void *, void *, size_t) = 0;

  /**
   * @brief Get the local handle object
   *
   * @param index
   * @return cublasHandle_t
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  virtual cublasHandle_t get_handle(int index) { return NULL; }

  /**
   * @brief Get the handle object
   *
   * @return cublasHandle_t
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  virtual cublasHandle_t get_handle() { return NULL; }

  /**
   * @brief matrix data memory malloc
   *
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  virtual void dev_malloc(_TYPE **, size_t, size_t) = 0;

  /**
   * @brief arraylist data memory malloc
   *
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  virtual void dev_malloc(_TYPE **, size_t) = 0;

  /**
   * @brief free memory
   *
   * @version 0.1
   * @author senh (ssh_9711@163.com)
   * @date 2022-03-23
   * @copyright Copyright (c) 2022
   */
  virtual void dev_free(void *) = 0;
};
