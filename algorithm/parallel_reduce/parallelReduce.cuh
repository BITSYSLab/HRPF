/*
 * @Author: your name
 * @Date: 2022-03-06 11:06:35
 * @LastEditTime: 2022-03-07 22:06:10
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_12\algorithm\parallel_reduce\parallel_recude.h
 */
#pragma once
#include "framework/problem.h"
#include "framework/task.h"
#include "common/gpu_device.h"
#include "framework/framework.h"
#include "datastructture/arraylist.h"
#include "datastructture/reduce_value.h"
#include "tool/helper.h"
#include <initializer_list>
// #include <thrust/reduce.h>
// #include <thrust/device_ptr.h>
// #include <thrust/system/cuda/execution_policy.h>
#include "test_return.cuh"
#include <thrust/functional.h>
#include <bitset>
#include <sys/time.h>
// #include "parallelReduce.tpp"

template<class Value_t, class Op_t>
struct ReduceData_t :public Basedata_t
{
    /* data *///size_t num_out,size_t num_in,
public:
    using ValueType = Value_t;
    using OperType = Op_t;

public:
    ReduceData_t(size_t s, size_t e,   
        ArrayList* buf, Value_t initial_v, Op_t oper)
            : buffer(buf), initial(initial_v)
    {
        start = s;
        end = e;
        m_oper = oper;
        ret = new RecudeCon<Value_t>(nullptr);
    }

public:
    ArrayList* buffer;
    size_t start;
    size_t end;
    Op_t m_oper;
    Value_t initial;
    RecudeCon<Value_t>* ret;
};

template<class Value_t, class Op_t>
class ReduceProblem: public Problem{
public:
    using ValueType = Value_t;
    using OperType = Op_t;
    ReduceData_t<Value_t, Op_t>* data_;

public:
    std::vector<Problem*> split() override;
    void merge(std::vector<Problem*>& subproblems) override;
    bool mustRunBaseCase();
    bool canRunBaseCase(int index);

public:
    ReduceProblem(ReduceData_t<Value_t, Op_t>* m_data, Function _cf, Function _gf, Problem* par) {
        data = m_data;
        data_ = m_data;
        cpu_func = _cf;
        gpu_func = _gf;
        parent = par;
        m_mask = std::bitset<T_SIZE>("100000");
    }
    virtual ~ReduceProblem() {
        if(data != nullptr){
            delete data;
            data = nullptr;
        }
    }

    void CPU_FUNC();
    void GPU_FUNC();
    void Input()  {}
    void Output()  {}
    void IO(Basedata_t* m_data);

};

// template<class Data_t>
// void cpu_func_(Data_t* data);
// template<class Data_t>
// void gpu_func_(Data_t* data);

// template<class Op_t, class T>
// void parallel_reduce(size_t start, size_t end, ArrayList* buf, 
//     T &init_value, Op_t oper);


template<class Op_t, class T>
inline void parallel_reduce(size_t start, size_t end, ArrayList* buf, 
    T init_value, Op_t oper) {
    // buf->access(Runtime::get_instance().get_gpu(), MemAccess::R);
    ReduceData_t<T, Op_t> *data = new ReduceData_t<T, Op_t>(start, end, buf, init_value, oper);
    auto problem = new ReduceProblem<T, Op_t>(data, nullptr, nullptr, nullptr); 
    Framework::solve(problem, "BBBBBBBBB");
    delete data->ret;
    delete problem;
}


template<class Value_t, class Op_t>
bool ReduceProblem<Value_t, Op_t>::mustRunBaseCase() {
    auto d = (ReduceData_t<Value_t, Op_t>*)data;
    return (d->end - d->start) <= 256;
}

template<class Value_t, class Op_t>
bool ReduceProblem<Value_t, Op_t>::canRunBaseCase(int index) {
    return m_mask[index] == 1;
}

template<class Value_t, class Op_t>
void ReduceProblem<Value_t, Op_t>::IO(Basedata_t* m_data) {
    auto m_d = (ReduceData_t<Value_t, Op_t>*)m_data;
    // int i = 0
    // for(int i = 0; i < m_d->n_in_buffer; ++i){
    //     inputAsc(m_d->buffer[i], MemAccess::R);
    // }
    inputAsc(m_d->buffer);
    // for(; i < m_d->buffer.size(); ++i){
    //     outputAsc(m_d->out_buffer[i], MemAccess::W);
    // }
    // outputAsc(m_d->ret);
}

template<class Value_t, class Op_t>
std::vector<Problem*> ReduceProblem<Value_t, Op_t>::split() {
    auto m_d = (ReduceData_t<Value_t, Op_t>*)data;
    auto buffer = m_d->buffer;
   
    ArrayList* left, *right;
    buffer->build_childs();
    left = buffer->get_child(0);
    right = buffer->get_child(1);
    size_t mid = (m_d->start + m_d->end) / 2;
    std::vector<Problem*> result(2);
    result[0] = new ReduceProblem<Value_t, Op_t>(new ReduceData_t<Value_t, Op_t>(0, mid-m_d->start,  
         left, m_d->initial, m_d->m_oper), nullptr, nullptr, this);//m_d->n_out_buffer,m_d->n_in_buffer,
    
    result[1] = new ReduceProblem<Value_t, Op_t>(new ReduceData_t<Value_t, Op_t>(0, m_d->end-mid,  
        right, m_d->initial, m_d->m_oper), nullptr, nullptr, this);//m_d->n_out_buffer, m_d->n_in_buffer,

    return result;
}

template<class Value_t, class Op_t>
void ReduceProblem<Value_t, Op_t>::merge(std::vector<Problem*>& subproblems){
    auto d = (ReduceData_t<Value_t, Op_t>*)data;
    int size = subproblems.size();
    auto sub_data1 = (ReduceData_t<Value_t, Op_t>*)((ReduceProblem<Value_t, Op_t>*)subproblems[0]->data);
    auto sub_data2 = (ReduceData_t<Value_t, Op_t>*)((ReduceProblem<Value_t, Op_t>*)subproblems[1]->data);
    // this->device = Runtime::get_instance().get_cpu();
    // subproblems[0]->device = this->device;
    // subproblems[1]->device = this->device;
    // input(sub_data1->ret);
    // input(sub_data2->ret);
    // output(d->ret);
    Value_t *ret_d = nullptr;
    
    ret_d = d->ret->get_cdata();
    auto l = sub_data1->ret->get_cdata();
    auto r = sub_data2->ret->get_cdata();
    *ret_d = d->m_oper(*l, *r);
    
    delete sub_data1->ret;
    delete sub_data2->ret;
    // delete subproblems[0];
    // delete subproblems[1];
}


// template<class Data_t, class Op_t, class Value_t>
// void cpu_func_(Data_t* data, size_t length, Op_t oper, Value_t ini_v, Value_t* ret) {
//     // auto d = data;
//     // size_t start = d->start;
//     // size_t end = d->end;
//     // auto init_v = d->initial;
//     // // decltype(init_v) local_ret = init_v;
    
//     // auto v_data = d->buffer->get_cdata();
    
//     // // for(size_t i = start; i < end; ++i){
//     // //     local_ret = d->m_oper(local_ret, v_data[i]);
//     // // }

//     // auto m_ret = d->ret->get_cdata();
//     *ret = thrust::reduce(data, data + length,
//                             ini_v,
//                             oper);
// }

template<class Value_t, class Op_t>
void ReduceProblem<Value_t, Op_t>::CPU_FUNC() {
    // cpu_func_(data_);
    // struct timeval start1, end1;
    // gettimeofday(&start1, NULL);
    auto d = (ReduceData_t<Value_t, Op_t>*)data;
    size_t start = d->start;
    size_t end = d->end;
    Value_t init_v = d->initial;
    auto v_data = d->buffer->get_cdata();
    Value_t* m_ret = d->ret->get_cdata();
    Op_t oper = d->m_oper;

    // #pragma omp declare reduction (local_oper: Value_t : omp_out = oper(omp_out, omp_in)) \
    //     initializer(omp_priv=init_v) 
    
    // #pragma omp parallel for simd reduction(local_oper : init_v) num_threads(12)
    // for(int i = start; i < end; ++i){
    //     init_v = oper(init_v, v_data[i]);
    // }
    // *m_ret = init_v;
    //  gettimeofday(&end1, NULL);
    // double seconds = 1000 * (end1.tv_sec - start1.tv_sec) + 1.0e-3 * (end1.tv_usec - start1.tv_usec);
    // std::cout << "openmp:" <<seconds << std::endl;
    cpu_func_(v_data, end - start, oper, init_v, m_ret);
}   

template<class Value_t, class Op_t>
void ReduceProblem<Value_t, Op_t>::GPU_FUNC() {
    // gpu_func_(data_);
    // struct timeval start1, end1;
    // gettimeofday(&start1, NULL);
    auto d = (ReduceData_t<Value_t, Op_t>*)data;
    size_t start = d->start;
    size_t end = d->end;
    Value_t init_v = d->initial;
    auto v_data = d->buffer->get_gdata();
    Value_t* m_ret = d->ret->get_cdata();
    Op_t oper = d->m_oper;
    cudaStream_t m_stream = stream();
    // #pragma omp declare reduction (oper: Value_t : omp_out = oper(omp_out, omp_in)) \
    //     initializer(omp_priv=init_v) 

    // #pragma target data map(tofrom:init_v) map(to : v_data[start:end])
    // {
    //     #pragma omp target teams distribute parallel for simd reduction(oper:init_v)
    //     {
    //         for(int i = start; i < end; ++i){
    //             init_v = oper(init_v, v_data[i]);
    //         }
    //     }
    // }
    // *m_ret = init_v;
    // int size = end - start;
    // if(size)
    gpu_func_(v_data, end - start, oper, init_v, m_ret, m_stream);
    //  gettimeofday(&end1, NULL);
    // double seconds = 1000 * (end1.tv_sec - start1.tv_sec) + 1.0e-3 * (end1.tv_usec - start1.tv_usec);
    // std::cout << "gpu:" <<seconds << std::endl;
}  

static int blocks_required = 1;
static int threads_per_block = 1024;

// template<class Op_t, class T>
// __global__ void kernel(size_t s, size_t e, size_t chunk, _TYPE* data, 
//     T ini_v, Op_t oper, T* ret, size_t elemSize) {

//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     int start = s + tid * chunk;
//     int end = start+chunk < e ? start + chunk : e;

//     T local_v = ini_v;
//     for(size_t i = start; i < end; ++i){
//         local_v = oper(local_v, data[i]);
//     }
    
//     size_t size = elemSize;
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

//     // int chunk_size = 1;
//     // int size = end - start;
//     // if(size % (threads_per_block * chunk_size)) {
//     //     blocks_required = size / (threads_per_block * chunk_size) + 1;
//     // }
//     // else {
//     //     blocks_required = size / (threads_per_block * chunk_size);
//     // }

//     auto oper = d->m_oper;
//     auto v_data = d->buffer->get_gdata();
//     auto ret_data = d->ret->get_cdata();
//     // size_t memSize = blocks_required * threads_per_block * sizeof(decltype(init_v));
//     cudaStream_t m_stream = stream();
//     // *ret_data = thrust::reduce(thrust::cuda::par.on(m_stream),
//     //     thrust::device_pointer_cast(v_data+start), thrust::device_pointer_cast(v_data + end),
//     //     init_v, oper);
//     // kernel<decltype(oper), decltype(init_v)><<<blocks_required, threads_per_block, memSize, m_stream>>>(start, end, chunk_size, v_data, 
//     //     init_v, oper, ret_data, blocks_required * threads_per_block);
// }
