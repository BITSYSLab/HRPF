/*
 * @Author: your name
 * @Date: 2022-01-17 19:09:03
 * @LastEditTime: 2022-02-16 14:51:59
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\1num_meachine\HRPA_NEW\tool\helper.h
 */
#pragma once

#include "common/gpu_device.h"
#include "common/runtime.h"

cublasHandle_t handle();
cudaStream_t stream();
