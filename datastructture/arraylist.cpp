/*
 * @Author: sen
 * @Date: 2021-11-01 20:27:55
 * @LastEditTime: 2022-02-20 19:12:14
 * @LastEditors: Please set LastEditors
 * @Description: arraylist type defination
 * @FilePath: \git_file_graduate\HRPA\common\arraylist.cpp
 */
#pragma once

#include "arraylist.h"
#include "common/runtime.h"

void ArrayList::access(Device *d, MemAccess ma) {
  ArrayPair_t &data = get_pair(d);
  ArrayPair_t &other = get_other(d);

  if (childs_[0] != nullptr) {
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
        // std::cout << "non asc RE" << std::endl;
        copy_from(data.first, other.first, d);
        data.second = MemState::SHARED;
        other.second = MemState::SHARED;
      }
      break;
    }
  }
  // return data.first;
}
void ArrayList::accessAsc(Device *d, MemAccess ma) {
  ArrayPair_t &data = get_pair(d);
  ArrayPair_t &other = get_other(d);

  if (childs_[0] != nullptr) {
    for (auto &child : childs_) {
      child->access(d, ma);
    }
  } else {
    switch (ma) {
    case MemAccess::W:
      /* code */
      // std::cout << "W oper" << std::endl;
      data.second = MemState::EXCLUSIVE;
      other.second = MemState::INVALID;
      break;

    case MemAccess::R:
      if (other.second == MemState::EXCLUSIVE) {
        // std::cout << "RE" << std::endl;
        copy_from_asc(data.first, other.first, d);
        data.second = MemState::SHARED;
        other.second = MemState::SHARED;
      }
      break;
    }
  }
}
ArrayList::ArrayPair_t &ArrayList::get_pair(Device *d) {
  if (d->get_type() == DeviceType::CPU) {
    return cpu_pair_;
  }

  return gpu_pair_;
}

ArrayList::ArrayPair_t &ArrayList::get_other(Device *d) {
  if (d->get_type() == DeviceType::CPU) {
    return gpu_pair_;
  }
  return cpu_pair_;
}

ArrayList::ArrayList(size_t _len) : length_(_len) {
  auto &runtime = Runtime::get_instance();
  dh_c = runtime.get_cpu();
  dh_g = runtime.get_gpu();
  dh_c->dev_malloc(&(cpu_pair_.first), _len);
#if PARALLEL_FOR
  cudaHostGetDevicePointer((void **)&(gpu_pair_.first),
                           (void *)(cpu_pair_.first), 0);
#else
  dh_g->dev_malloc(&(gpu_pair_.first), _len);
#endif
  cpu_pair_.second = MemState::INVALID;
  gpu_pair_.second = MemState::INVALID;
  ld_ = _len;
  mallocd_ = true;
}

ArrayList::ArrayList(_TYPE *data, size_t len) : length_(len) {
  auto &runtime = Runtime::get_instance();
  dh_c = runtime.get_cpu();
  dh_g = runtime.get_gpu();
  cpu_pair_.first = data;
#if PARALLEL_FOR
  cudaHostGetDevicePointer((void **)&(gpu_pair_.first),
                           (void *)(cpu_pair_.first), 0);
#else
  dh_g->dev_malloc(&(gpu_pair_.first), len);
#endif
  cpu_pair_.second = MemState::EXCLUSIVE;
  gpu_pair_.second = MemState::INVALID;
  ld_ = len;
  mallocd_ = true;
}

ArrayList::ArrayList(ArrayList *_parent, size_t index) : parent_(_parent) {

  length_ = parent_->length() / 2;
  ld_ = _parent->get_ld();
  dh_c = _parent->dh_c;
  dh_g = _parent->dh_g;
  cpu_pair_.first = parent_->get_cdata(index % 2 * length_);
  cpu_pair_.second = parent_->get_cpu_pair().second;

  gpu_pair_.first = parent_->get_gdata(index % 2 * length_);
  gpu_pair_.second = parent_->get_gpu_pair().second;
  mallocd_ = false;
}

ArrayList::ArrayList(ArrayList *_parent, size_t index, size_t length)
    : parent_(_parent) {
  length_ = length;
  ld_ = _parent->get_ld();
  dh_c = _parent->dh_c;
  dh_g = _parent->dh_g;
  cpu_pair_.first = parent_->get_cdata(index);
  cpu_pair_.second = parent_->get_cpu_pair().second;

  gpu_pair_.first = parent_->get_gdata(index);
  gpu_pair_.second = parent_->get_gpu_pair().second;
}

ArrayList::~ArrayList() {
  if (childs_[0] != nullptr) {
    for (size_t i = 0; i < 2; ++i) {
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

size_t ArrayList::length() { return length_; }

void ArrayList::setLength(size_t lens) { length_ = lens; }

ArrayList *ArrayList::get_child(size_t index) { return childs_[index]; }

void ArrayList::build_childs() {
  if (childs_[0] != nullptr)
    return;
  for (size_t i = 0; i < 2; ++i) {
    childs_[i] = new ArrayList(this, i);
  }

  if (length_ % 2) {
    childs_[1]->setLength(length_ - childs_[0]->length());
  }
}

void ArrayList::build_xchilds(int index) {
  if (childs_[0] != nullptr)
    return;
  childs_[0] = new ArrayList(this, 0, index);
  childs_[0] = new ArrayList(this, index, length_ - index);
}

ArrayList::ArrayPair_t &ArrayList::get_cpu_pair() { return cpu_pair_; }

ArrayList::ArrayPair_t &ArrayList::get_gpu_pair() { return gpu_pair_; }

_TYPE *ArrayList::get_cdata() { return cpu_pair_.first; }

_TYPE *ArrayList::get_cdata(size_t _x) { return cpu_pair_.first + _x; }

_TYPE *ArrayList::get_gdata() { return gpu_pair_.first; }

_TYPE *ArrayList::get_gdata(size_t _x) { return gpu_pair_.first + _x; }

void ArrayList::copy_from(_TYPE *src, _TYPE *dst, Device *d) {
  d->dev_mem_put(src, dst, length_);
}
void ArrayList::copy_from_asc(_TYPE *src, _TYPE *dst, Device *d) {
  d->dev_mem_put_asc(src, dst, length_);
}
size_t ArrayList::get_ld() { return ld_; }