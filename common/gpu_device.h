/**
 * @file gpu_device.h
 * @author senh (ssh_9711@163.com)
 * @brief  gpu device class.
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
#include <condition_variable>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mutex>

//#include "framework/problem.h"
#include "framework/config.h"
#include <vector>

// #include "threadpool/threadpool.h"
/*Forward declaration*/
class Problem;
/*Forward declaration*/
class Task;

/**
 * @brief
 *     gpu device class declare.
 */
class GpuDevice : public Device {
public:
  /*cublas handle*/
  cublasHandle_t m_handle_;
  /*stream mutex*/
  std::mutex m_streams_mutex;
  /*stream cv*/
  std::condition_variable m_streams_cv;
  /*idle stream pool*/
  std::vector<cudaStream_t> m_idle_streams;
  /*thread local stream*/
  cudaStream_t m_curr_stream[T_SIZE];
  /*thread local cublas handle*/
  cublasHandle_t m_handle[T_SIZE];

public:
  /**
   * @brief Construct a new Gpu Device object
   *
   */
  GpuDevice() : Device(DeviceType::GPU) {
    if (STREAM_NUM_) {
      m_idle_streams.resize(STREAM_NUM_);
    }
    for (int i = 0; i < STREAM_NUM_; ++i) {
      cudaStreamCreateWithFlags(&m_idle_streams[i], cudaStreamNonBlocking);
      // cublasCreate(&m_handle[i]);
      // cublasSetStream(m_handle[i], m_idle_streams[i]);
    }
    // if(HANDLE_NUM_) cublasCreate(&m_handle_);
    for (int i = 0; i < HANDLE_NUM_; ++i) {
      cublasCreate(&m_handle[i]);
    }
    // pool.destroy();
  }

  /**
   * @brief Destroy the Gpu Device object
   *
   */
  ~GpuDevice() {
    for (int i = 0; i < STREAM_NUM_; ++i) {
      cudaStreamDestroy(m_idle_streams[i]);
      // cublasDestroy(m_handle[i]);
    }

    // if(HANDLE_NUM_) cublasDestroy(m_handle_);
    for (int i = 0; i < HANDLE_NUM_; ++i) {
      cublasDestroy(m_handle[i]);
    }
    // std::cout << "delete gpu..." << std::endl;
  }

public:
  /**
   * @brief Get the handle object
   *
   * @param index : thread index
   * @return cublasHandle_t
   */
  cublasHandle_t get_handle(int index) { return m_handle_; }

  /**
   * @brief Get the handle object
   *
   * @return cublasHandle_t
   */
  cublasHandle_t get_handle() { return m_handle_; }

  /**
   * @brief gpu memory malloc
   *
   * @param ptr    : device pointer
   * @param width  : matrix width
   * @param height : matrix height
   */
  void dev_malloc(_TYPE **ptr, size_t width, size_t height); // TODO done

  /**
   * @brief gpu memory malloc
   *
   * @param ptr    : arraylist pointer
   * @param length : array length
   */
  void dev_malloc(_TYPE **ptr, size_t length);

  /**
   * @brief release gpu memory
   *
   * @param ptr : device pointer
   */
  void dev_free(void *ptr); // TODO done

  /**
   * @brief matrix data copy
   *
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
   *
   * @param dst    : copy dst pointer
   * @param src    : copy src pointer
   * @param length : length of array
   */
  void dev_mem_put(void *dst, void *src, size_t length);

  /**
   * @brief matrix data async copy
   *
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
   *
   * @param dst    : copy dst pointer
   * @param src    : copy src pointer
   * @param length : length of array
   */
  void dev_mem_put_asc(void *dst, void *src, size_t length);

  /**
   * @brief recycle local stream
   *
   * @param stream : cuda stream
   */
  void recycle_stream(cudaStream_t stream);

  /**
   * @brief parallel for sync
   *
   */
  void synchronize();
  // static void CUDART_CB  call_back(cudaStream_t stream, cudaError_t status,
  // void* userData); static void CUDART_CB  call_back_mer(cudaStream_t stream,
  // cudaError_t status, void* userData);
};

/**
 * @brief
 *      call back func arg.
 */
struct CallBackData {
  /*gpu device*/
  GpuDevice *gpu;
  /*problem*/
  Problem *task;
  /*problem num*/
  int m_size;
  /*task*/
  Task *t_task;

  /**
   * @brief Construct a new Call Back Data object
   *
   * @param d     : gpu
   * @param t     : problem
   * @param size  : problem num
   * @param task_ : task
   */
  CallBackData(GpuDevice *d, Problem *t, int size = 0, Task *task_ = NULL)
      : gpu(d), task(t), m_size(size), t_task(task_) {}
};

/**
 * @brief cuda stream call_back func \
 *        per problem async exec -> p->par atmoic_sub(deps)
 * @param stream : cuda stream
 * @param status : null
 * @param userData : call back data
 */
void CUDART_CB call_back(cudaStream_t stream, cudaError_t status,
                         void *userData);

/**
 * @brief cuda stream call_back func \
 *        async exec a task problem in split/merge oper -> sync
 * @param stream : cuda stream
 * @param status : null
 * @param userData : call back data
 */
void CUDART_CB call_back_mer(cudaStream_t stream, cudaError_t status,
                             void *userData);

/**
 * @brief cuda stream call_back func \
 *        async exec a single problem in merge oper -> sync
 * @param stream : cuda stream
 * @param status : null
 * @param userData : call back data
 */
void CUDART_CB call_back_single(cudaStream_t stream, cudaError_t status,
                                void *userData);