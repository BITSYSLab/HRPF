/*
 * @Author: your name
 * @Date: 2022-02-16 14:01:29
 * @LastEditTime: 2022-02-16 14:27:09
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_NEW_VERSION\HRPA_NEW\algorithm\strassen_problem\cuAdd.h
 */
#pragma once

#include <cuda_runtime.h>

#define _TYPE double
void sumMatrixInplace(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, const double p_one);
void sumMatrix(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, cudaStream_t stream);
void subMatrix(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, cudaStream_t stream);
void gemm(_TYPE* MatA, _TYPE* MatB, _TYPE* MatC, int dim, int lda, int ldb, int ldc, cudaStream_t stream);
