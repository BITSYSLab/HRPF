#pragma once

#include "parallelReduce.cuh"

// static int blocks_required = 1;
// static int threads_per_block = 1024;

// template<class Op_t, class T>
// __global__ void kernel(size_t s, size_t e, size_t chunk, _TYPE* data, 
//     T ini_v, Op_t oper, T* ret) {

//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int start = s + tid * chunk;
//     int end = start+chunk < e ? start + chunk : e;

//     T local_v = ini_v;
//     for(size_t i = start; i < end; ++i){
//         local_v = oper(local_v, data[i]);
//     }
    
//     size_t size = blocks_required * threads_per_block;
//     extern __shared__ T r[];
//     r[tid] = local_v; 
//     __syncthreads();
//     for (int i = size/2; i>0; i/=2) { //uniform
//         if (tid < i)
//             r[tid] += r[tid+i];
//         __syncthreads();
//     }

//     if(tid == 0) {
//         *ret = r[0];
//     }

// }

// template<class Data_t>
// void gpu_func_(Data_t* data) {
//     auto d = data;
//     size_t start = d->start;
//     size_t end = d->end;
//     auto init_v = d->initial;

//     int chunk_size = 1;
//     int size = end - start;
//     if(size % (threads_per_block * chunk_size)) {
//         blocks_required = size / (threads_per_block * chunk_size) + 1;
//     }
//     else {
//         blocks_required = size / (threads_per_block * chunk_size);
//     }

//     auto oper = d->m_oper;
//     _TYPE *v_data = d->buffer->get_gdata();
//     auto ret_data = d->ret->get_gdata();
//     size_t memSize = blocks_required * threads_per_block * sizeof(decltype(init_v));
//     kernel<decltype(oper), decltype(init_v)><<<blocks_required, threads_per_block, memSize>>>(start, end, chunk_size, v_data, 
//         init_v, oper, ret_data);
// }