#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <optional>

#define LogSize 8

template <typename T>
class SharedTaskQueue
{

  static_assert(std::is_pointer_v<T>, "T must be a pointer type");

  constexpr static int64_t BufferSize = int64_t{1} << LogSize;
  constexpr static int64_t BufferMask = (BufferSize - 1);

  static_assert((BufferSize >= 2) && ((BufferSize & (BufferSize - 1)) == 0));

  alignas(128) std::atomic<int64_t> _top{0};
  alignas(128) std::atomic<int64_t> _bottom{0};
  alignas(128) std::atomic<T> _buffer[BufferSize];

public:
  SharedTaskQueue() = default;
  ~SharedTaskQueue() = default;

  bool empty() const noexcept;
  size_t size() const noexcept;
  constexpr size_t capacity() const noexcept;

  // Any-thread push to the "back" of the deque
  bool push_back(T item);

  // Any-thread pop from the "back" of the deque
  T pop_back();

  // Any-thread pop from the "front" of the deque (steal)
  T pop_front();

  // Owner-only push to the "front" of the deque (for distribution)
  bool push_front(T item);

  // Owner-only batch push to the "front" of the deque
  template <typename InputIt>
  bool push_front_batch(InputIt first, InputIt last);
};

#include "common/shared_task_queue.h"

template <typename T>
bool SharedTaskQueue<T>::empty() const noexcept {
  int64_t t = _top.load(std::memory_order_relaxed);
  int64_t b = _bottom.load(std::memory_order_relaxed);
  return b <= t;
}

template <typename T>
size_t SharedTaskQueue<T>::size() const noexcept {
  int64_t t = _top.load(std::memory_order_relaxed);
  int64_t b = _bottom.load(std::memory_order_relaxed);
  return static_cast<size_t>(b >= t ? b - t : 0);
}

template <typename T>
constexpr size_t SharedTaskQueue<T>::capacity() const noexcept {
  return static_cast<size_t>(BufferSize);
}

template <typename T>
bool SharedTaskQueue<T>::push_front(T item) {
  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_acquire);

  if ((b - t) >= BufferSize) {
    return false; // Full
  }
  
  _buffer[b & BufferMask].store(item, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  _bottom.store(b + 1, std::memory_order_release);
  return true;
}

template <typename T>
template <typename InputIt>
bool SharedTaskQueue<T>::push_front_batch(InputIt first, InputIt last) {
  
  const auto n = std::distance(first, last);

  if (n == 0) {
    return true;
  }
  
  int64_t b = _bottom.load(std::memory_order_relaxed);
  int64_t t = _top.load(std::memory_order_acquire);

  // Check if there's enough space for 'n' items.
  if (BufferSize - (b - t) < n) {
    return false; // Not enough space
  }

  int64_t i = b;
  for (auto it = first; it != last; ++it, ++i) {
    _buffer[i & BufferMask].store(*it, std::memory_order_relaxed);
  }
  std::atomic_thread_fence(std::memory_order_release);
  
  _bottom.store(b + n, std::memory_order_release);
  
  return true;
}

template <typename T>
T SharedTaskQueue<T>::pop_front() {
  int64_t b = _bottom.fetch_sub(1, std::memory_order_acq_rel);
  int64_t t = _top.load(std::memory_order_acquire);
  
  T item{nullptr};

  if (t < b) {
    item = _buffer[(b - 1) & BufferMask].load(std::memory_order_relaxed);
    if (t == b - 1) {
      if (!_top.compare_exchange_strong(t, t + 1, 
                                       std::memory_order_seq_cst, 
                                       std::memory_order_relaxed)) {
        item = nullptr;
      }
    }
  }
  else {
    return nullptr;
  }

  return item;
}

template <typename T>
T SharedTaskQueue<T>::pop_back() {
  int64_t t = _top.load(std::memory_order_acquire);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = _bottom.load(std::memory_order_acquire);
  
  T item{nullptr};

  if(t < b) {
    item = _buffer[t & BufferMask].load(std::memory_order_relaxed);
    if(!_top.compare_exchange_strong(t, t + 1,
                                     std::memory_order_seq_cst,
                                     std::memory_order_relaxed)) {
      return nullptr;
    }
  }
  return item;
}


template <typename T>
bool SharedTaskQueue<T>::push_back(T item) {
  int64_t t = _top.load(std::memory_order_relaxed);

  while (true) {
    int64_t b = _bottom.load(std::memory_order_acquire);

    if (b - (t - 1) > BufferSize) {
      return false; // Queue is full.
    }
    if (_top.compare_exchange_weak(t, t - 1,
                                   std::memory_order_acq_rel,
                                   std::memory_order_relaxed)) {
      _buffer[(t - 1) & BufferMask].store(item, std::memory_order_release);
      return true;
    }
  }
}