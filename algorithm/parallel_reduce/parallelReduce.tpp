/*
 * @Author: your name
 * @Date: 2022-03-06 15:45:22
 * @LastEditTime: 2022-03-07 22:09:46
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_12\algorithm\parallel_reduce\parallelReduce.cpp
 */
#pragma once

// #include "parallelReduce.h"

// template<class Value_t, class Op_t>
// bool ReduceProblem<Value_t, Op_t>::mustRunBaseCase() {
//     auto d = (ReduceData_t<Value_t, Op_t>*)data;
//     return (d->end - d->start) <= 128;
// }

// template<class Value_t, class Op_t>
// bool ReduceProblem<Value_t, Op_t>::canRunBaseCase(int index) {
//     return m_mask[index] == 1;
// }

// template<class Value_t, class Op_t>
// void ReduceProblem<Value_t, Op_t>::IO(Basedata_t* m_data) {
//     auto m_d = (ReduceData_t<Value_t, Op_t>*)m_data;
//     // int i = 0
//     // for(int i = 0; i < m_d->n_in_buffer; ++i){
//     //     inputAsc(m_d->buffer[i], MemAccess::R);
//     // }
//     inputAsc(m_d->buffer);
//     // for(; i < m_d->buffer.size(); ++i){
//     //     outputAsc(m_d->out_buffer[i], MemAccess::W);
//     // }
//     outputAsc(m_d->ret);
// }

// template<class Value_t, class Op_t>
// std::vector<Problem*> ReduceProblem<Value_t, Op_t>::split() {
//     auto m_d = (ReduceData_t<Value_t, Op_t>*)data;
//     auto buffer = m_d->buffer;
//     // size_t buffer_size = m_d->buffer.size();

//     // std::vector<ArrayList*> left(buffer_size), right(buffer_size);

//     // for(int i = 0; i < buffer_size; ++i) {
//     //     buffer[i]->build_childs();
//     //     left[i] = buffer[i]->get_child(0);
//     //     right[i] = buffer[i]->get_child(1);
//     // }
//     ArrayList* left, *right;
//     buffer->build_childs();
//     left = buffer->get_child(0);
//     right = buffer->get_child(1);
//     size_t mid = (m_d->start + m_d->end) / 2;
//     std::vector<Problem*> result(2);
//     result[0] = new ReduceProblem<Value_t, Op_t>(new ReduceData_t<Value_t, Op_t>(0, m_d->mid-m_d->start,  
//          left, m_d->initial, m_d->m_oper), cpu_func_, gpu_func_, this);//m_d->n_out_buffer,m_d->n_in_buffer,
    
//     result[1] = new ReduceProblem<Value_t, Op_t>(new ReduceData_t<Value_t, Op_t>(0, m_d->end-m_d->mid,  
//         left, m_d->initial, m_d->m_oper), cpu_func_, gpu_func_, this);//m_d->n_out_buffer, m_d->n_in_buffer,
// }

// template<class Value_t, class Op_t>
// void ReduceProblem<Value_t, Op_t>::merge(std::vector<Problem*>& subproblems){
//     auto d = (ReduceData_t<ValueType, OperType>*)data;
//     int size = subproblems.size();
//     auto sub_data1 = (ReduceData_t<ValueType, OperType>*)((ReduceProblem<Value_t, Op_t>*)subproblems[0]->data);
//     auto sub_data2 = (ReduceData_t<ValueType, OperType>*)((ReduceProblem<Value_t, Op_t>*)subproblems[1]->data);
//     this->device = Runtime::get_instance().get_cpu();
//     (ReduceProblem<Value_t, Op_t>*)subproblems[0]->device = this->device;
//     (ReduceProblem<Value_t, Op_t>*)subproblems[1]->device = this->device;
//     input(sub_data1->ret);
//     input(sub_data2->ret);
//     output(d->ret);
//     ValueType *ret_d = nullptr;
    
//     ret_d = d->ret->get_cdata();
//     auto l = sub_data1->ret->get_cdata();
//     auto r = sub_data2->ret->get_cdata();
//     *ret_d = d->m_oper(*l, *r);
    
//     delete sub_data1->ret;
//     delete sub_data2->ret;
    
// }


// template<class Data_t>
// void cpu_func_(Data_t* data) {
//     auto d = data;
//     size_t start = d->start;
//     size_t end = d->end;
//     auto init_v = d->initial;
//     decltype(init_v) local_ret = init_v;
    
//     _TYPE *v_data = d->buffer->get_cdata();
    
//     for(size_t i = start; i < end; ++i){
//         local_ret = d->m_oper(local_ret, v_data[i]);
//     }

//     auto m_ret = d->ret->get_cdata();
//     *m_ret = local_ret;
// }

// template<class Value_t, class Op_t>
// void ReduceProblem<Value_t, Op_t>::CPU_FUNC() {
//     cpu_func_(data_);
// }   

// template<class Value_t, class Op_t>
// void ReduceProblem<Value_t, Op_t>::GPU_FUNC() {
//     gpu_func_(data_);
// }  

