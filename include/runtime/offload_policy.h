#pragma once
#include <cstddef>

struct OffloadPolicy {
    // 当一个 base case 任务的 global_ordinal_id_ 是此值的整数倍时，
    // 它成为一个卸载候选。例如，如果设为 30，则 ID 为 0, 30, 60, ... 的任务
    // 会被考虑卸载。
    // 这代表了你所说的“GPU/CPU性能差距倍数”。
    // 0 表示禁用此策略。
    size_t high_water_mark = 0;

    // 当任务被卸载到CPU上时，应当再分多少层来让CPU worker更忙碌
    size_t additional_depth = 4;

    // 问题规模阈值。当一个 base case 任务的 problem_size 小于此值时，
    // 它成为一个卸载候选。
    // 0 表示禁用此策略。
    size_t problem_size_threshold = 0;

    // default constructor, won't offload
    OffloadPolicy() = default;

    OffloadPolicy(size_t mark, size_t depth = 4, size_t threshold = 0)
        : high_water_mark(mark), additional_depth(depth), problem_size_threshold(threshold) {}
};