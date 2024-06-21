/**
 * @file matrix.h
 * @author senh (ssh_9711@163.com)
 * @brief  matrix data declare
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
#include "common/runtime.h"
#include <array>
#include <cassert>
#include <vector>

class Matrix {
public:
  /*data---state*/
  typedef typename std::pair<_TYPE *, MemState> MatrixPair_t;

private:
  /*child matrix*/
  std::vector<Matrix *> childs_;
  /*parent matrix*/
  Matrix *parent_ = nullptr;
  /*x dimension*/
  size_t xdim_ = 0;
  /*y dimension*/
  size_t ydim_ = 0;
  /*matrix ld*/
  size_t ld_;
  /*cpu data pair*/
  MatrixPair_t cpu_pair_;
  /*gpu data pair*/
  MatrixPair_t gpu_pair_;
  /*matrix malloc flag*/
  bool mallocd_ = false;
  Matrix(const Matrix &);  // refuse left value copy
  Matrix(const Matrix &&); // refuse right value copy
  // Matrix& opreator=(const Matrix&); //refuse =

public:
  /*cpu device*/
  Device *dh_c;
  /*gpu device*/
  Device *dh_g;

public:
  /**
   * @brief Get the pair object
   *
   * @param d              : device
   * @return MatrixPair_t& : d--pair
   */
  MatrixPair_t &get_pair(Device *d);

  /**
   * @brief Get the other object
   *
   * @param d              : device
   * @return MatrixPair_t& : d--pair
   */
  MatrixPair_t &get_other(Device *d);

  /**
   * @brief Get the cpu pair object
   *
   * @return MatrixPair_t&
   */
  MatrixPair_t &get_cpu_pair();

  /**
   * @brief Get the gpu pair object
   *
   * @return MatrixPair_t&
   */
  MatrixPair_t &get_gpu_pair();

  /**
   * @brief Construct a new Matrix object
   *
   * @param _xd  : width
   * @param _yd  : height
   */
  Matrix(size_t _xd, size_t _yd);

  /**
   * @brief Construct a new Matrix object
   *
   * @param data : cpu data
   * @param _xd  : width
   * @param _yd  : height
   */
  Matrix(_TYPE *data, size_t _xd, size_t _yd);

  /**
   * @brief Construct a new Matrix object
   *         from parent --- even split
   * @param _parent : parent matrix
   * @param index   : index
   */
  Matrix(Matrix *_parent, size_t index);

  /**
   * @brief Construct a new Matrix object
   *            col split or row split
   * @param _parent : parent data
   * @param index   : index
   * @param c       : row / col flag
   */
  Matrix(Matrix *_parent, size_t index, char c);

  /**
   * @brief Construct a new Matrix object
   *
   * @param _parent : parent
   * @param x       :  xidx
   * @param y       :  yidx
   * @param block   : col block
   */
  Matrix(Matrix *_parent, size_t x, size_t y, size_t block);

  /**
   * @brief Construct a new Matrix object
   *
   * @param _parent : parent
   * @param x       : xidx
   * @param block   : row block
   */
  Matrix(Matrix *_parent, size_t x, size_t block);

  /**
   * @brief Destroy the Matrix object
   *
   */
  ~Matrix();

  /**
   * @brief Set the xdim object
   *
   * @param xdim : row dimension lens
   */
  void set_xdim(size_t xdim);

  /**
   * @brief Set the ydim object
   *
   * @param ydim : col dimension lens
   */
  void set_ydim(size_t ydim);

  /**
   * @brief Get the xdim
   *
   * @return size_t
   */
  size_t get_xdim();

  /**
   * @brief Get the ydim
   *
   * @return size_t
   */
  size_t get_ydim();

  /**
   * @brief Get the ld
   *
   * @return size_t
   */
  size_t get_ld();

  /**
   * @brief  even split childs matrix
   *
   */
  void build_childs();

  /**
   * @brief row even split
   *
   */
  void build_rchilds();

  /**
   * @brief col even split
   *
   */
  void build_cchilds();

  /**
   * @brief col blocks split
   *
   * @param blocks : col blocks
   */
  void build_childs(std::vector<int> &blocks);

  /**
   * @brief row blocks split
   *
   * @param blocks : row blocks
   */
  void build_childrs(std::vector<int> &blocks);

  /**
   * @brief Get the child
   *
   * @param index     : child index
   * @return Matrix*
   */
  Matrix *get_child(size_t index);

  /**
   * @brief  manage data state
   *
   */
  void access(Device *, MemAccess);

  /**
   * @brief  async data state
   *
   * @param d   : device
   * @param ma  : read/write
   */
  void accessAsc(Device *d, MemAccess ma);

  /**
   * @brief    data copy
   *
   * @param src  : copy to
   * @param dst  : copy from
   * @param d    : device
   */
  void copy_from(_TYPE *src, _TYPE *dst, Device *d);

  /**
   * @brief   async data copy
   *
   * @param src  : copy to
   * @param dst  : copy from
   * @param d    : device
   */
  void copy_from_asc(_TYPE *src, _TYPE *dst, Device *d);

  /**
   * @brief Get the cpu data object
   *
   * @return _TYPE*
   */
  _TYPE *get_cdata();

  /**
   * @brief Get the cpudata object
   *
   * @param _x      : row idx
   * @param _y      : col idx
   * @return _TYPE*
   */
  _TYPE *get_cdata(size_t _x, size_t _y);

  /**
   * @brief Get the cpudata object
   *
   * @param _index  : data index
   * @return _TYPE*
   */
  _TYPE *get_cdata(size_t _index);

  /**
   * @brief Get the gpudata object
   *
   * @return _TYPE*
   */
  _TYPE *get_gdata();

  /**
   * @brief Get the gpudata object
   *
   * @param _x      : row idx
   * @param _y      : col idx
   * @return _TYPE*
   */
  _TYPE *get_gdata(size_t _x, size_t _y);

  /**
   * @brief Get the gpudata object
   *
   * @param _index  : data index
   * @return _TYPE*
   */
  _TYPE *get_gdata(size_t _index);
};
