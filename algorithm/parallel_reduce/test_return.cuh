#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

template<class Data, class Op_t, class Value_t, class Stream_t>
void gpu_func_(Data* data, size_t length, Op_t oper, Value_t ini_v, Value_t* ret, Stream_t stream){
    *ret = thrust::reduce(thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(data), thrust::device_pointer_cast(data + length),
        ini_v, oper);
}

template<class Data, class Op_t, class Value_t>
void cpu_func_(Data* data, size_t length, Op_t oper, Value_t ini_v, Value_t* ret) {
    *ret = thrust::reduce(data, data + length,
                            ini_v,
                            oper);
    // #pragma omp declare reduction (c_oper: double : omp_out = oper(omp_out, omp_in)) \
    //     initializer(omp_priv=0) 
    // #pragma omp parallel for num_threads(8) reduction(+:ini_v)
    // for(int i = 0; i < length; ++i) {
    //     ini_v = oper(ini_v, data[i]);
    // } 
    // *ret = ini_v;
}