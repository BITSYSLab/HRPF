cmake_minimum_required(VERSION 3.18)
project(hrpf LANGUAGES CXX CUDA)

# ====================================================================
#                      PERFORMANCE OPTIMIZATIONS
# ====================================================================

#   Run cmake with: cmake -DCMAKE_BUILD_TYPE=Release ..
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # C++ Compiler Flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math -g")

    # CUDA Compiler Flags
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")

    message(STATUS "Release build. Applying performance flags: -O3, native arch, fast-math.")
endif()

# 2. Enable Link-Time Optimization (LTO) for global optimizations
include(CheckIPOSupported)
check_ipo_supported(RESULT result)
if(result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "Link-Time Optimization (LTO) is enabled.")
else()
    message(WARNING "Link-Time Optimization (LTO) is not supported.")
endif()

# ====================================================================

# -------------------------------
# 设置 C++ 和 CUDA 标准
# -------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -------------------------------
# 设置输出目录
# -------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# -------------------------------
# 设置 CUDA 编译标志
# -------------------------------
set(CMAKE_CUDA_ARCHITECTURES 89)
# CMAKE_CUDA_FLAGS 已经包含了 release/debug 标志，所以用 CMAKE_CUDA_FLAGS 追加
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC --extended-lambda")

# -------------------------------
# 搜索源文件（递归查找 src 下的 .cpp 和 .cu 文件）
# -------------------------------
file(GLOB_RECURSE CPP_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)
# file(GLOB_RECURSE CUDA_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cu)
# set(ALL_SOURCES ${CPP_SOURCES} ${CUDA_SOURCES})
set(ALL_SOURCES ${CPP_SOURCES})


# -------------------------------
# 生成共享库 libhrpf.so
# -------------------------------
add_library(hrpf SHARED ${ALL_SOURCES})
set_target_properties(hrpf PROPERTIES
    OUTPUT_NAME "hrpf"
    SUFFIX ".so"
)

# -------------------------------
# 设置链接器标志（如 --no-as-needed）
# -------------------------------
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")

# -------------------------------
# 链接外部库 (现代CMake方式)
# -------------------------------
find_package(CUDAToolkit REQUIRED)

# 3. 为你的目标链接库。
target_link_libraries(hrpf PUBLIC
    CUDA::cudart          # CUDA Runtime
    CUDA::cublas
    pthread               # POSIX 线程
    m                     # 数学库
    dl                    # 动态链接库
)

# 4. (可选但推荐) 如果你的 C++ 文件也需要包含项目内的头文件
target_include_directories(hrpf PUBLIC ${CMAKE_SOURCE_DIR}/include)