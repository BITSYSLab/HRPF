/*
 * @Author: your name
 * @Date: 2021-12-01 16:55:40
 * @LastEditTime: 2021-12-01 20:50:54
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_NEW\tool\cuMerge.h
 */
#pragma once

//#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/merge.h>
#include <thrust/system/cuda/execution_policy.h>
#define _TYPE double

extern "C++" void merge_sort(_TYPE* first, _TYPE* second, int len);
void gsort(_TYPE* data, int len, cudaStream_t stream);
void gmerge(_TYPE* first, _TYPE* second, _TYPE* dst, int len, cudaStream_t stream);
void hsort(_TYPE* data, int len);
void gsort(_TYPE* data, int len);
void hmerge(_TYPE* first, _TYPE* second, _TYPE* dst, int len);
void gmerge(_TYPE* first, _TYPE* second, _TYPE* dst, int len);
