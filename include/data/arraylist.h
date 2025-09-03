#pragma once

#include "data/data.h"

template <typename T>
class ArrayList : public BaseData<T>
{
public:
    // 引入基类的 get_cdata() 方法
    using BaseData<T>::get_cdata;
    using BaseData<T>::get_gdata;

    explicit ArrayList(size_t _len) : BaseData<T>(_len * sizeof(T)), length_(_len), stride_(1) {}

    ArrayList(T *data, size_t len) : BaseData<T>(data, len * sizeof(T)), length_(len), stride_(1) {}

    ArrayList(T *data, size_t len, size_t stride) : BaseData<T>(data, len * sizeof(T)), length_(len), stride_(stride) {}

    ArrayList(ArrayList *_parent, size_t start_offset, size_t child_length)
        : BaseData<T>(_parent, _parent->get_cdata(start_offset), _parent->get_gdata(start_offset)),
          length_(child_length), stride_(_parent->stride_) {}

    size_t length() const { return length_; }

    void set_length(size_t len) { length_ = len; }

    size_t get_stride() const { return stride_; }

    T *get_cdata(size_t offset) { return this->cpu_pair_.first + offset * stride_; }

    T *get_gdata(size_t offset) { return this->gpu_pair_.first + offset * stride_; }

    /* special partition when n_slices == 2 */
    void partition() {
        if (length_ <= 1) return;

        size_t mid = length_ / 2;
        this->children_.push_back(std::make_unique<ArrayList<T>>(this, 0, mid));
        this->children_.push_back(std::make_unique<ArrayList<T>>(this, mid, length_ - mid));
    }

    void partition(size_t offset)
    {
        if (!this->children_.empty())
            return;
        
        if (offset > 0) {
            this->children_.push_back(std::make_unique<ArrayList<T>>(this, 0, offset));
        }
        if (offset < length_) {
            this->children_.push_back(std::make_unique<ArrayList<T>>(this, offset, length_ - offset));
        }
    }

    /* offsets' length == n_slices + 1 and should always start with 0 and end with length_ */
    void partition(int n_slices, const std::vector<size_t> &offsets)
    {
        if (!this->children_.empty() || n_slices <= 1 || offsets.size() != n_slices + 1 || offsets.back() != length_)
        {
            return;
        }

        for (int i = 0; i < n_slices; ++i)
        {
            size_t start_offset = offsets[i];
            size_t end_offset = offsets[i + 1];

            if (end_offset <= start_offset || end_offset > length_)
            {
                this->children_.clear();
                return;
            }

            size_t child_len = end_offset - start_offset;
            this->children_.push_back(std::make_unique<ArrayList<T>>(this, start_offset, child_len));
        }
    }

protected:
    void copy_from(T *src, T *dst, Device *d, cudaStream_t stream)
    {
        d->dev_mem_put_asc(src, dst, length_ * sizeof(T), stream);
        cudaStreamSynchronize(stream);
    }

    void copy_from_asc(T *src, T *dst, Device *d, cudaStream_t stream)
    {
        d->dev_mem_put_asc(src, dst, length_ * sizeof(T), stream);
    }

private:
    size_t length_;
    size_t stride_;
};
