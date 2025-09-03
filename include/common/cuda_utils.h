#pragma once

#include "cuda_runtime.h"
#include <stdexcept>
#include <string>
#include <iostream>

inline void cuda_check_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::string error_msg = "CUDA Error: " + std::string(cudaGetErrorString(err)) +
                                " in " + std::string(file) +
                                " at line " + std::to_string(line);
        throw std::runtime_error(error_msg);
    }
}

#define CUDA_CHECK(call) cuda_check_error(call, __FILE__, __LINE__)