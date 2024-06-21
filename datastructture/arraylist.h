/**
 * @file arraylist.h
 * @author senh (ssh_9711@163.com)
 * @brief  arraylist datastructure
 * @version 0.1
 * @date 2022-03-24
 *
 * @copyright Copyright (c) 2022
 *
 */
#pragma once

#include "common/data.h"
#include "common/device.h"
#include "common/enum.h"
#include <array>
#include <cassert>

/**
 * @brief 1-D array class declare
 *
 */
class ArrayList {
public:
  /*data---state pair_t*/
  typedef typename std::pair<_TYPE *, MemState> ArrayPair_t;

  /**
   * @brief      data copy
   *
   * @param src : src pointer
   * @param dst : dst pointer
   * @param d   : device
   */
  void copy_from(_TYPE *src, _TYPE *dst, Device *d);

  /**
   * @brief      async data copy
   *
   * @param src  : src pointer
   * @param dst  : dst pointer
   * @param d    : device
   */
  void copy_from_asc(_TYPE *src, _TYPE *dst, Device *d);
  // Device* get_device();

  /**
   * @brief Get the cpudata object
   *
   * @return _TYPE*
   */
  _TYPE *get_cdata();

  /**
   * @brief Get the cpudata object
   *
   * @param _x      :  child index
   * @return _TYPE*
   */
  _TYPE *get_cdata(size_t _x);

  /**
   * @brief Get the gpudata object
   *
   * @return _TYPE*
   */
  _TYPE *get_gdata();

  /**
   * @brief Get the gpudata object
   *
   * @param _x      : child index
   * @return _TYPE*
   */
  _TYPE *get_gdata(size_t _x);

public:
  /*cpu device*/
  Device *dh_c;
  /*gpu device*/
  Device *dh_g;

private:
  /*child matrix*/
  std::array<ArrayList *, 2> childs_ = {{nullptr, nullptr}};
  /*parent matrix*/
  ArrayList *parent_ = nullptr;
  /*array length*/
  size_t length_ = 0;
  /*ld*/
  size_t ld_;
  /*malloc or not*/
  bool mallocd_ = false;
  /*cpu data pair*/
  ArrayPair_t cpu_pair_;
  /*gpu data pair*/
  ArrayPair_t gpu_pair_;
  ArrayList(const ArrayList &);
  ArrayList(const ArrayList &&);

public:
  /**
   * @brief Get the pair object
   *
   * @param d             : device
   * @return ArrayPair_t&
   */
  ArrayPair_t &get_pair(Device *d);

  /**
   * @brief Get the other object
   *
   * @param d             : device
   * @return ArrayPair_t&
   */
  ArrayPair_t &get_other(Device *d);

  /**
   * @brief Get the cpu pair object
   *
   * @return ArrayPair_t&
   */
  ArrayPair_t &get_cpu_pair();

  /**
   * @brief Get the gpu pair object
   *
   * @return ArrayPair_t&
   */
  ArrayPair_t &get_gpu_pair();

  /**
   * @brief Construct a new Array List object
   *
   * @param _len : length of array
   */
  ArrayList(size_t _len);

  /**
   * @brief Construct a new Array List object
   *
   * @param data : cpu array data
   * @param len  : length
   */
  ArrayList(_TYPE *data, size_t len);

  /**
   * @brief Construct a new Array List object
   *        even split
   * @param _parent : parent array
   * @param index   : child index
   */
  ArrayList(ArrayList *_parent, size_t index);

  /**
   * @brief Construct a new Array List object
   *
   * @param _parent : parent array
   * @param index   : split point
   * @param length  : child array length
   */
  ArrayList(ArrayList *_parent, size_t index, size_t length);

  /**
   * @brief Destroy the Array List object
   *
   */
  ~ArrayList();

  /**
   * @brief get array length
   *
   * @return size_t
   */
  size_t length();

  /**
   * @brief Set the Length object
   *
   * @param lens : array length
   */
  void setLength(size_t lens);

  /**
   * @brief Get the ld object
   *
   * @return size_t
   */
  size_t get_ld();

  /**
   * @brief creates childs array(even)
   *
   */
  void build_childs();

  /**
   * @brief create childs array based on index
   *
   */
  void build_xchilds(int index);

  /**
   * @brief Get the child object
   *
   * @param index       : child idx
   * @return ArrayList*
   */
  ArrayList *get_child(size_t index);

  /**
   * @brief manage data state
   *
   */
  void access(Device *, MemAccess);

  /**
   * @brief async manage data state
   *
   * @param d  : device
   * @param ma : read/write mode
   */
  void accessAsc(Device *d, MemAccess ma);
};
