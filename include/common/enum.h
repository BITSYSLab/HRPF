#pragma once

enum class DeviceType {
    CPU,
    GPU
};

enum class TaskStrategy {
    BFS,
    DFS
};

enum class MemState {
    INVALID,
    SHARED,
    EXCLUSIVE
};

enum class MemAccess {
    R,
    W,
    RW
};

enum class WorkMode {
    ACTIVE,
    HYBRID
};

enum class WorkerMode {
    CPU_ONLY,
    GPU_ONLY,
    HETERO
};

#ifndef _TYPE
#define _TYPE double
#endif