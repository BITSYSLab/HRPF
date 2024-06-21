/*
 * @Author: sen
 * @Date: 2021-11-01 16:44:47
 * @LastEditTime: 2022-02-20 09:58:37
 * @LastEditors: Please set LastEditors
 * @Description: matrix type
 * @FilePath: \git_file_graduate\HRPA\common\matrix.cpp
 */

#pragma once

#include "matrix.h"

void Matrix::access(Device *d, MemAccess ma) {
  MatrixPair_t &data = get_pair(d);
  MatrixPair_t &other = get_other(d);

  if (!childs_.empty()) {
    for (auto &child : childs_) {
      child->access(d, ma);
    }
  } else {
    switch (ma) {
    case MemAccess::W:
      /* code */
      // if(other.second == MemState::EXCLUSIVE){
      //     copy_from(data.first, other.first, d);
      // }
      data.second = MemState::EXCLUSIVE;
      other.second = MemState::INVALID;
      break;

    case MemAccess::R:
      if (other.second == MemState::EXCLUSIVE) {
        copy_from(data.first, other.first, d);
        data.second = MemState::SHARED;
        other.second = MemState::SHARED;
      }
      break;
    }
    // switch(data.second){
    //     case MemState::INVALID: {
    //         switch(other.second){
    //             case MemState::INVALID:
    //                 assert(ma == MemAccess::W);
    //                 data.second = MemState::EXCLUSIVE;
    //                 other.second = MemState::INVALID;
    //                 break;
    //             case MemState::SHARED:
    //                 assert(0);
    //                 break;
    //             case MemState::EXCLUSIVE:
    //                 if(ma == MemAccess::W){
    //                     data.second = MemState::EXCLUSIVE;
    //                     other.second = MemState::INVALID;
    //                 }else{

    //                     data.first->copy_from(other.first);
    //                     data.second = MemState::SHARED;
    //                     other.second = MemState::SHARED;
    //                 }
    //                 break;
    //         }
    //         break;
    //     }

    //     case MemState::SHARED:
    //     {
    //         switch(other.second){
    //             case MemState::INVALID:
    //                 assert(0);
    //                 break;
    //             case MemState::SHARED:
    //                 if(ma == MemAccess::W){
    //                     data.second = MemState::EXCLUSIVE;
    //                     other.second = MemState::INVALID;
    //                 }
    //                 break;
    //             case MemState::EXCLUSIVE:
    //                 assert(0);
    //                 break;
    //         }
    //         break;
    //     }

    //     case MemState::EXCLUSIVE:
    //     {
    //         switch(other.second){
    //             case MemState::INVALID:
    //                 break;
    //             case MemState::EXCLUSIVE:
    //                 assert(0);
    //                 break;
    //             case MemState::SHARED:
    //                 assert(0);
    //                 break;
    //         }
    //         break;
    //     }
    // }
  }
  // return data.first;
}

void Matrix::accessAsc(Device *d, MemAccess ma) {
  MatrixPair_t &data = get_pair(d);
  MatrixPair_t &other = get_other(d);

  if (!childs_.empty()) {
    for (auto &child : childs_) {
      child->access(d, ma);
    }
  } else {
    switch (ma) {
    case MemAccess::W:
      /* code */
      data.second = MemState::EXCLUSIVE;
      other.second = MemState::INVALID;
      break;

    case MemAccess::R:
      if (other.second == MemState::EXCLUSIVE) {
        copy_from_asc(data.first, other.first, d);
        data.second = MemState::SHARED;
        other.second = MemState::SHARED;
      }
      break;
    }
  }
}

Matrix::MatrixPair_t &Matrix::get_cpu_pair() { return cpu_pair_; }

Matrix::MatrixPair_t &Matrix::get_gpu_pair() { return gpu_pair_; }

Matrix::MatrixPair_t &Matrix::get_pair(Device *d) {
  if (d->get_type() == DeviceType::CPU) {
    return cpu_pair_;
  }
  return gpu_pair_;
}

Matrix::MatrixPair_t &Matrix::get_other(Device *d) {
  if (d->get_type() == DeviceType::CPU) {
    return gpu_pair_;
  }
  return cpu_pair_;
}

Matrix::Matrix(size_t _xd, size_t _yd) : xdim_(_xd), ydim_(_yd) {
  auto &runtime = Runtime::get_instance();
  dh_c = runtime.get_cpu();
  dh_g = runtime.get_gpu();
  dh_c->dev_malloc(&(cpu_pair_.first), xdim_, ydim_);
#if PARALLEL_FOR
  cudaHostGetDevicePointer((void **)&(gpu_pair_.first),
                           (void *)(cpu_pair_.first), 0);
#else
  dh_g->dev_malloc(&(gpu_pair_.first), xdim_, ydim_);
#endif

  cpu_pair_.second = MemState::INVALID;
  gpu_pair_.second = MemState::INVALID;
  ld_ = xdim_;
  mallocd_ = true;
}

Matrix::Matrix(_TYPE *data, size_t _xd, size_t _yd) : xdim_(_xd), ydim_(_yd) {
  auto &runtime = Runtime::get_instance();
  dh_c = runtime.get_cpu();
  dh_g = runtime.get_gpu();
  cpu_pair_.first = data;
#if PARALLEL_FOR
  cudaHostGetDevicePointer((void **)&(gpu_pair_.first),
                           (void *)(cpu_pair_.first), 0);
#else
  dh_g->dev_malloc(&(gpu_pair_.first), xdim_, ydim_);
#endif
  cpu_pair_.second = MemState::EXCLUSIVE;
  gpu_pair_.second = MemState::INVALID;
  ld_ = xdim_;
  mallocd_ = true;
}

Matrix::Matrix(Matrix *_parent, size_t index) : parent_(_parent) {
  // auto& runtime = Runtime::get_instance();
  xdim_ = parent_->get_xdim() / 2;
  ydim_ = parent_->get_ydim() / 2;
  dh_c = _parent->dh_c;
  dh_g = _parent->dh_g;

  ld_ = _parent->get_ld();
  cpu_pair_.first = _parent->get_cdata((index % 2) * xdim_, index / 2 * ydim_);
  cpu_pair_.second = parent_->get_cpu_pair().second;
  gpu_pair_.first = _parent->get_gdata((index % 2) * xdim_, index / 2 * ydim_);
  gpu_pair_.second = parent_->get_gpu_pair().second;
}

Matrix::Matrix(Matrix *_parent, size_t index, char c) : parent_(_parent) {
  // auto& runtime = Runtime::get_instance();
  if (c == 'r') {
    xdim_ = parent_->get_xdim() / 2;
    ydim_ = parent_->get_ydim();
  } else {
    ydim_ = parent_->get_ydim() / 2;
    xdim_ = parent_->get_xdim();
  }

  dh_c = _parent->dh_c;
  dh_g = _parent->dh_g;

  ld_ = _parent->get_ld();
  cpu_pair_.first = _parent->get_cdata((index % 2) * xdim_, index / 2 * ydim_);
  cpu_pair_.second = parent_->get_cpu_pair().second;
  gpu_pair_.first = _parent->get_gdata((index % 2) * xdim_, index / 2 * ydim_);
  gpu_pair_.second = parent_->get_gpu_pair().second;
}

Matrix::Matrix(Matrix *_parent, size_t x, size_t y, size_t block)
    : parent_(_parent) {

  xdim_ = parent_->get_xdim();
  ydim_ = parent_->get_ydim();
  ydim_ = (y + block) < ydim_ ? block : ydim_ - y;
  dh_c = _parent->dh_c;
  dh_g = _parent->dh_g;
  ld_ = _parent->get_ld();
  cpu_pair_.first = _parent->get_cdata(x, y);
  cpu_pair_.second = parent_->get_cpu_pair().second;
  gpu_pair_.first = _parent->get_gdata(x, y);
  gpu_pair_.second = parent_->get_gpu_pair().second;
  mallocd_ = false;
}

Matrix::Matrix(Matrix *_parent, size_t x, size_t block) : parent_(_parent) {

  xdim_ = parent_->get_xdim();
  ydim_ = parent_->get_ydim();
  xdim_ = (x + block) < xdim_ ? block : xdim_ - x;
  dh_c = _parent->dh_c;
  dh_g = _parent->dh_g;
  ld_ = _parent->get_ld();
  cpu_pair_.first = _parent->get_cdata(x, 0);
  cpu_pair_.second = parent_->get_cpu_pair().second;
  gpu_pair_.first = _parent->get_gdata(x, 0);
  gpu_pair_.second = parent_->get_gpu_pair().second;
  mallocd_ = false;
}

size_t Matrix::get_xdim() { return xdim_; }

size_t Matrix::get_ydim() { return ydim_; }

void Matrix::set_xdim(size_t xdim) { xdim_ = xdim; }

void Matrix::set_ydim(size_t ydim) { ydim_ = ydim; }

size_t Matrix::get_ld() { return ld_; }

void Matrix::build_childs() {
  // if(childs_[0] != nullptr) return;
  for (size_t i = 0; i < 4; ++i) {
    // childs_[i] = new Matrix(this, i);
    childs_.emplace_back(new Matrix(this, i));
  }

  if (xdim_ % 2 || ydim_ % 2) {
    /*set right dimension value*/
    childs_[2]->set_ydim(ydim_ - childs_[0]->get_ydim());
    childs_[1]->set_xdim(xdim_ - childs_[0]->get_xdim());
    childs_[3]->set_ydim(childs_[2]->get_ydim());
    childs_[3]->set_xdim(childs_[1]->get_xdim());
  }
}

void Matrix::build_rchilds() {
  childs_.emplace_back(new Matrix(this, 0, 'r'));
  childs_.emplace_back(new Matrix(this, 1, 'r'));

  if (xdim_ % 2) {
    childs_[1]->set_xdim(xdim_ - childs_[0]->get_xdim());
  }
}

void Matrix::build_cchilds() {
  childs_.emplace_back(new Matrix(this, 0, 'c'));
  childs_.emplace_back(new Matrix(this, 2, 'c'));

  if (ydim_ % 2) {
    childs_[1]->set_ydim(ydim_ - childs_[1]->get_ydim());
  }
}

void Matrix::build_childs(std::vector<int> &blocks) {
  if (!childs_.empty())
    return;
  int size = blocks.size();
  int init_block = 0;
  for (int i = 0; i < size; ++i) {
    childs_.emplace_back(new Matrix(this, 0, init_block, blocks[i]));
    init_block += blocks[i];
  }
}

void Matrix::build_childrs(std::vector<int> &blocks) {
  if (!childs_.empty())
    return;
  int size = blocks.size();
  int init_block = 0;
  for (int i = 0; i < size; ++i) {
    childs_.emplace_back(new Matrix(this, init_block, blocks[i]));
    init_block += blocks[i];
  }
}

Matrix *Matrix::get_child(size_t index) { return childs_[index]; }

Matrix::~Matrix() {
  // if(childs_[0] != nullptr){
  if (!childs_.empty()) {
    int size = childs_.size();
    for (size_t i = 0; i < size; ++i) {
      delete childs_[i];
      childs_[i] = nullptr;
    }
  }

  if (mallocd_) {
#if PARALLEL_FOR
    dh_c->dev_free(cpu_pair_.first);
#else
    dh_c->dev_free(cpu_pair_.first);
    dh_g->dev_free(gpu_pair_.first);
#endif
  }
}

_TYPE *Matrix::get_cdata() { return cpu_pair_.first; }

_TYPE *Matrix::get_gdata() { return gpu_pair_.first; }

_TYPE *Matrix::get_cdata(size_t _x, size_t _y) {
  return cpu_pair_.first + _x + _y * ld_;
}

_TYPE *Matrix::get_gdata(size_t _x, size_t _y) {
  return gpu_pair_.first + _x + _y * ld_;
}

_TYPE *Matrix::get_cdata(size_t _index) { return cpu_pair_.first + _index; }

_TYPE *Matrix::get_gdata(size_t _index) { return gpu_pair_.first + _index; }

void Matrix::copy_from(_TYPE *src, _TYPE *dst, Device *d) {
  d->dev_mem_put(src, ld_ * sizeof(_TYPE), dst, ld_ * sizeof(_TYPE),
                 xdim_ * sizeof(_TYPE), ydim_);
}

void Matrix::copy_from_asc(_TYPE *src, _TYPE *dst, Device *d) {
  d->dev_mem_put_asc(src, ld_ * sizeof(_TYPE), dst, ld_ * sizeof(_TYPE),
                     xdim_ * sizeof(_TYPE), ydim_);
}
