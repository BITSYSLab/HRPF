// /*
//  * @Author: your name
//  * @Date: 2022-03-06 10:41:35
//  * @LastEditTime: 2022-03-07 14:08:47
//  * @LastEditors: Please set LastEditors
//  * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
//  * @FilePath: \git_file_graduate\HRPA_12\datastructture\reduce_value.cpp
//  */
// #pragma once

// #include "reduce_value.h"
// #include "common/runtime.h"

// template<class Value_t>
// void RecudeCon<Value_t>::access(Device *d, MemAccess ma) {
//     ReducePair_t &data = get_pair(d);
//     ReducePair_t &other = get_other(d);

//     switch (ma)
//     {
//     case MemAccess::W:
//         /* code */
//         data.second = MemState::EXCLUSIVE;
//         other.second = MemState::INVALID;
//         break;

//     case MemAccess::R:
//         if(other.second == MemState::EXCLUSIVE){
//         copy_from(data.first, other.first, d);
//         data.second = MemState::SHARED;
//         other.second = MemState::SHARED;
//         }
//         break;
//     }    
// }

// template<class Value_t>
// void RecudeCon<Value_t>::accessAsc(Device *d, MemAccess ma) {
//     ReducePair_t &data = get_pair(d);
//     ReducePair_t &other = get_other(d);
    
//     switch (ma)
//     {
//     case MemAccess::W:
//         /* code */
//         data.second = MemState::EXCLUSIVE;
//         other.second = MemState::INVALID;
//         break;

//     case MemAccess::R:
//         if(other.second == MemState::EXCLUSIVE){
//         copy_from(data.first, other.first, d);
//         data.second = MemState::SHARED;
//         other.second = MemState::SHARED;
//         }
//         break;
//     }    
// }

// template<class Value_t>
// typename RecudeCon<Value_t>::ReducePair_t& RecudeCon<Value_t>::get_pair(Device *d) {
//   if (d->get_type() == DeviceType::CPU)
//   {
//     return cpu_pair_;
//   }  
  
//   return gpu_pair_;
// }

// template<class Value_t>
// typename RecudeCon<Value_t>::ReducePair_t& RecudeCon<Value_t>::get_other(Device *d) {
//   if (d->get_type() == DeviceType::CPU)
//   {
//     return gpu_pair_;
//   }  
//   return cpu_pair_;
// }

// template<class Value_t>
// typename RecudeCon<Value_t>::ReducePair_t& RecudeCon<Value_t>::get_cpu_pair() {
//   return cpu_pair_;
// }

// template<class Value_t>
// typename RecudeCon<Value_t>::ReducePair_t& RecudeCon<Value_t>::get_gpu_pair() {
//   return gpu_pair_;
// }

// template<class Value_t>
// void RecudeCon<Value_t>::copy_from(Value_t* src, Value_t* dst, Device* d) {
//     d->dev_mem_put(src, dst, 1);
// }

// template<class Value_t>
// void RecudeCon<Value_t>::copy_from_asc(Value_t* src, Value_t* dst, Device* d) {
//     d->dev_mem_put_asc(src, dst, 1);
// }

// template<class Value_t>
// RecudeCon<Value_t>::RecudeCon(Value_t *data) {
//     auto &runtime = Runtime::get_instance();
//     dh_c = runtime.get_cpu();
//     dh_g = runtime.get_gpu();
//     dh_c->dev_malloc(&(cpu_pair_.first), 1);
//     dh_g->dev_malloc(&(gpu_pair_.first), 1);
//     cpu_pair_.second = MemState::INVALID;
//     gpu_pair_.second = MemState::INVALID;
//     mallocd_ = true;
// }

// template<class Value_t>
// RecudeCon<Value_t>::~RecudeCon() {
//     if(mallocd_) {
//       dh_c->dev_free(cpu_pair_.first);
//       dh_g->dev_free(gpu_pair_.first);
//     }
// }

// template<class Value_t>
// Value_t* RecudeCon<Value_t>::get_cdata() {
//     return cpu_pair_.first;
// }

// template<class Value_t>
// Value_t* RecudeCon<Value_t>::get_gdata() {
//     return gpu_pair_.first;
// }