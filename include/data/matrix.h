#pragma once

#include "data/data.h"
#include <stdexcept>

template <typename T>
class Matrix : public BaseData<T>
{
public:
    using BaseData<T>::get_cdata;
    using BaseData<T>::get_gdata;

    explicit Matrix(size_t rows, size_t cols)
        : BaseData<T>(rows * cols * sizeof(T)), rows_(rows), cols_(cols), ld_(cols)
    {
        if (rows == 0 || cols == 0)
        {
            throw std::invalid_argument("Matrix dimensions cannot be zero.");
        }
    }

    Matrix(T *data, size_t rows, size_t cols)
        : BaseData<T>(data, rows * cols * sizeof(T)), rows_(rows), cols_(cols), ld_(cols)
    {
        if (rows == 0 || cols == 0)
        {
            throw std::invalid_argument("Matrix dimensions cannot be zero.");
        }
    }

        Matrix(T *cpu_data, T* gpu_data, size_t rows, size_t cols, DeviceType type)
        : BaseData<T>(cpu_data, gpu_data, rows * cols * sizeof(T), type), rows_(rows), cols_(cols), ld_(cols)
    {
        if (rows == 0 || cols == 0)
        {
            throw std::invalid_argument("Matrix dimensions cannot be zero.");
        }
    }

    Matrix(T *data, size_t rows, size_t cols, size_t ld)
        : BaseData<T>(data, rows * ld * sizeof(T)), rows_(rows), cols_(cols), ld_(ld)
    {
        if (rows == 0 || cols == 0)
        {
            throw std::invalid_argument("Matrix dimensions cannot be zero.");
        }
    }

    Matrix(Matrix<T> *parent, size_t start_row, size_t start_col, size_t num_rows, size_t num_cols)
        : BaseData<T>(parent,
                      parent->get_cdata(start_row, start_col),
                      parent->get_gdata(start_row, start_col)),
          rows_(num_rows), cols_(num_cols), ld_(parent->ld_) {}

    size_t get_rows() const { return rows_; }
    size_t get_cols() const { return cols_; }
    size_t get_ld() const { return ld_; }

    T *get_cdata(size_t row, size_t col) { return this->cpu_pair_.first + row * ld_ + col; }
    T *get_gdata(size_t row, size_t col) { return this->gpu_pair_.first + row * ld_ + col; }

    /* partition the matrix into 4 slices */
    void partition_by_quadrants()
    {
        if (!this->children_.empty())
            return;

        size_t row_mid = rows_ / 2;
        size_t col_mid = cols_ / 2;

        // TL
        this->children_.push_back(std::make_unique<Matrix>(this, 0, 0, row_mid, col_mid));
        // TR
        this->children_.push_back(std::make_unique<Matrix>(this, 0, col_mid, row_mid, cols_ - col_mid));
        // BL
        this->children_.push_back(std::make_unique<Matrix>(this, row_mid, 0, rows_ - row_mid, col_mid));
        // BR
        this->children_.push_back(std::make_unique<Matrix>(this, row_mid, col_mid, rows_ - row_mid, cols_ - col_mid));
    }

    /* horizon slices */
    void partition_by_rows(const std::vector<size_t> &row_offsets)
    {
        if (!this->children_.empty() || row_offsets.size() < 2 || row_offsets.front() != 0 || row_offsets.back() != rows_)
        {
            return;
        }
        for (size_t i = 0; i < row_offsets.size() - 1; ++i)
        {
            size_t start_row = row_offsets[i];
            size_t end_row = row_offsets[i + 1];
            if (end_row <= start_row)
                continue;
            this->children_.push_back(std::make_unique<Matrix>(this, start_row, 0, end_row - start_row, cols_));
        }
    }

    /* vertical slices */
    void partition_by_cols(const std::vector<size_t> &col_offsets)
    {
        if (!this->children_.empty() || col_offsets.size() < 2 || col_offsets.front() != 0 || col_offsets.back() != cols_)
        {
            return;
        }
        for (size_t i = 0; i < col_offsets.size() - 1; ++i)
        {
            size_t start_col = col_offsets[i];
            size_t end_col = col_offsets[i + 1];
            if (end_col <= start_col)
                continue;
            this->children_.push_back(std::make_unique<Matrix>(this, 0, start_col, rows_, end_col - start_col));
        }
    }

protected:
    void copy_from(T *src, T *dst, Device *d, cudaStream_t stream)
    {
        d->dev_mem_put_asc(src, ld_ * sizeof(T), dst, ld_ * sizeof(T),
                           cols_ * sizeof(T), rows_, stream);
        cudaStreamSynchronize(stream);
    }

    void copy_from_asc(T *src, T *dst, Device *d, cudaStream_t stream)
    {
        d->dev_mem_put_asc(src, ld_ * sizeof(T), dst, ld_ * sizeof(T),
                           cols_ * sizeof(T), rows_, stream);
    }

private:
    size_t rows_;
    size_t cols_;
    size_t ld_;
};
