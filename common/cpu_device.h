/**
 * @file cpu_device.h
 * @author senh (ssh_9711@163.com)
 * @brief  cpu device class
 * @version 0.1
 * @date 2022-03-23
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include "device.h"
#include "enum.h"
#include <cassert>
#include <cuda_runtime.h>

/**
 * @brief
 *     cpu device class declare.
 */
class CpuDevice : public Device {
public:
  /**
   * @brief Construct a new Cpu Device object
   *
   */
  CpuDevice() : Device(DeviceType::CPU){};

public:
  /**
   * @brief cpu memory malloc
   *
   * @param ptr    : host pointer
   * @param width  : matrix width
   * @param height : matrix height
   */
  void dev_malloc(_TYPE **ptr, size_t width, size_t height); // TODO

  /**
   * @brief cpu memory malloc
   *
   * @param ptr    : arraylist pointer
   * @param length : array length
   */
  void dev_malloc(_TYPE **ptr, size_t length);

  /**
   * @brief release cpu memory
   *
   * @param ptr : host pointer
   */
  void dev_free(void *ptr); // TODO

  /**
   * @brief matrix data copy
   *          host func
   * @param dst    : copy dst pointer
   * @param dpitch : dst pitch
   * @param src    : copy src pointer
   * @param spitch : src pitch
   * @param width  : matrix width
   * @param height : matrix height
   */
  void dev_mem_put(void *dst, size_t dpitch, void *src, size_t spitch,
                   size_t width, size_t height);

  /**
   * @brief arraylist data copy
   *          host func
   * @param dst    : copy dst pointer
   * @param src    : copy src pointer
   * @param length : length of array
   */
  void dev_mem_put(void *dst, void *src, size_t length);

  /**
   * @brief matrix data async copy
   *          host func
   * @param dst    : copy dst pointer
   * @param dpitch : dst pitch
   * @param src    : copy src pointer
   * @param spitch : src pitch
   * @param width  : matrix width
   * @param height : matrix height
   */
  void dev_mem_put_asc(void *dst, size_t dpitch, void *src, size_t spitch,
                       size_t width, size_t height);

  /**
   * @brief arraylist data async copy
   *          host func
   * @param dst    : copy dst pointer
   * @param src    : copy src pointer
   * @param length : length of array
   */
  void dev_mem_put_asc(void *dst, void *src, size_t length);
};