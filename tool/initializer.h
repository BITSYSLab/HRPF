/*
 * @Author: your name
 * @Date: 2021-11-30 09:40:37
 * @LastEditTime: 2022-02-14 13:41:41
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置:
 * https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_NEW\tool\initializer.h
 */
#pragma once

#include "common/enum.h"
#include "common/runtime.h"
#include "datastructture/arraylist.h"
#include "datastructture/matrix.h"
#include <float.h>
#include <math.h>
#include <memory.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/**
 * @brief
 * INITIALIZE THE SQUARE MATRIX WITH RANDOM VALUE;
 */
void initialize(int dim, Matrix *ha, Matrix *hb) {
  // srand48(time(NULL));
  auto &runtime = Runtime::get_instance();
  auto cpu = runtime.get_cpu();
  (ha)->access(cpu, MemAccess::W);
  (hb)->access(cpu, MemAccess::W);
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      *(ha->get_cdata(i, j)) = (_TYPE)(1);
      *(hb->get_cdata(i, j)) = (_TYPE)(1);
    }
  }
  auto gpu = runtime.get_cpu();
  // ha->access(gpu, MemAccess::R);
  // hb->access(gpu, MemAccess::R);
}

void initialize(int dim, Matrix *ha) {
  // srand48(time(NULL));
  auto &runtime = Runtime::get_instance();
  auto cpu = runtime.get_cpu();
  (ha)->access(cpu, MemAccess::W);
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      *(ha->get_cdata(i, j)) = (_TYPE)(rand() % 100);
    }
  }
}

void initialize(ArrayList *data, int length) {
  srand48(time(NULL));
  auto &runtime = Runtime::get_instance();
  auto cpu = runtime.get_cpu();
  (data)->access(cpu, MemAccess::W);
  for (int i = 0; i < length; ++i) {
    *(data->get_cdata(i)) = (_TYPE)(rand() % 100);
  }
}
