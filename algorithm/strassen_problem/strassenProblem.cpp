/*
 * @Author: your name
 * @Date: 2021-11-24 15:17:53
 * @LastEditTime: 2022-02-22 20:18:10
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_NEW\algorithm\strassenProblem.cpp
 */
#pragma once

#include "strassenProblem.h"
#include "tool/helper.h"
#include "cuAdd.h"

StrassenProblem::StrassenProblem(Basedata_t* d, Function cf, Function gf, Problem* par) {
    data = d;
    cpu_func = cf;
    gpu_func = gf;
    parent = par;
    m_mask = std::bitset<T_SIZE>("10");
}

void StrassenProblem::Input() {
    auto d = (StrassenData_t*)data;

    input(d->ha);
    input(d->hb);
}

void StrassenProblem::Output() {
    auto d = (StrassenData_t*)data;
    output(d->hc);
}

void StrassenProblem::IO(Basedata_t* m_data) {
    auto d = (StrassenData_t*)m_data;

    inputAsc(d->ha);
    inputAsc(d->hb);
    outputAsc(d->hc);
}

bool StrassenProblem::canRunBaseCase(int index){
    return m_mask[index] == 1;
}


/***************************
* end condition
***************************/
bool StrassenProblem::mustRunBaseCase() {
    return !flag;
}

/***************************
* split operation
***************************/
std::vector<Problem*> StrassenProblem::split()  {
    auto m_data = (StrassenData_t*)data;
    m_data->ha->build_childs();
    auto a11 = m_data->ha->get_child(0);
    auto a12 = m_data->ha->get_child(2);
    auto a21 = m_data->ha->get_child(1);
    auto a22 = m_data->ha->get_child(3);

    m_data->hb->build_childs();
    auto b11 = m_data->hb->get_child(0);
    auto b12 = m_data->hb->get_child(2);
    auto b21 = m_data->hb->get_child(1);
    auto b22 = m_data->hb->get_child(3);
    
    size_t dim = m_data->ha->get_xdim() / 2;
    
    //18-temporary matrix 
    Matrix* s1 = new Matrix(dim, dim);
    Matrix* s2 = new Matrix(dim, dim);
    Matrix* s3 = new Matrix(dim, dim);
    Matrix* s4 = new Matrix(dim, dim);
    Matrix* t1 = new Matrix(dim, dim);
    Matrix* t2 = new Matrix(dim, dim);
    Matrix* t3 = new Matrix(dim, dim);
    Matrix* t4 = new Matrix(dim, dim);
    Matrix* p1 = new Matrix(dim, dim);
    Matrix* p2 = new Matrix(dim, dim);
    Matrix* p3 = new Matrix(dim, dim);
    Matrix* p4 = new Matrix(dim, dim);
    Matrix* p5 = new Matrix(dim, dim);
    Matrix* p6 = new Matrix(dim, dim);
    Matrix* p7 = new Matrix(dim, dim);

    //non-recursive operation
    Strassen *task_s3 = new Strassen(new StrassenData_t(a11, a21, s3), cpu_sub, gpu_sub, this);
    Strassen *task_t3 = new Strassen(new StrassenData_t(b22, b12, t3), cpu_sub, gpu_sub, this);
    Task*  t_task1 = new Task({task_s3, task_t3});
    t_task1->run(this, 'g');
    // std::cout << "t_task_1" << std::endl;
    Strassen *task_s1 = new Strassen(new StrassenData_t(a21, a22, s1), cpu_add, gpu_add, this);
    Strassen *task_s2 = new Strassen(new StrassenData_t(s1, a11, s2), cpu_sub, gpu_sub, this);
    Strassen *task_s4 = new Strassen(new StrassenData_t(a12, s2, s4), cpu_sub, gpu_sub, this);
    Task* t_task2 = new Task({task_s1, task_s2, task_s4});
    t_task2->run(this, 'g');
    // std::cout << "t_task2" << std::endl;
    Strassen *task_t1 = new Strassen(new StrassenData_t(b12, b11, t1), cpu_sub, gpu_sub, this);
    Strassen *task_t2 = new Strassen(new StrassenData_t(b22, t1, t2), cpu_sub, gpu_sub, this);
    Strassen *task_t4 = new Strassen(new StrassenData_t(t2, b21, t4), cpu_sub, gpu_sub, this);
    Task* t_task3 = new Task({
        task_t1, task_t2, task_t4});
    t_task3->run(this, 'g');
    // std::cout << "t_task3" << std::endl;

    //recursive operation
    std::vector<Problem*> result(7);    
    result[0] = new Strassen(new StrassenData_t(s3, t3, p7), cpu_mul, gpu_mul, this);
    result[1] = new Strassen(new StrassenData_t(s1, t1, p5), cpu_mul, gpu_mul, this);
    result[2] = new Strassen(new StrassenData_t(s2, t2, p6), cpu_mul, gpu_mul, this);
    result[3] = new Strassen(new StrassenData_t(s4, b22, p3), cpu_mul, gpu_mul, this);
    result[4] = new Strassen(new StrassenData_t(a11, b11, p1), cpu_mul, gpu_mul, this);
    result[5] = new Strassen(new StrassenData_t(a12, b21, p2), cpu_mul, gpu_mul, this);
    result[6] = new Strassen(new StrassenData_t(a22, t4, p4), cpu_mul, gpu_mul, this);

    // delete task_s1;
    // delete task_s2;
    // delete task_s3;
    // delete task_s4;
    // delete task_t1;
    // delete task_t2;
    // delete task_t3;
    // delete task_t4;
    return result;
}

/***************************
* merge operation
***************************/
void StrassenProblem::merge(std::vector<Problem*>& subproblems)  {
    // std::cout << "merge" << std::endl;
    auto m_data = (StrassenData_t*)data;
    m_data->hc->build_childs();
    auto c11 = m_data->hc->get_child(0);
    auto c12 = m_data->hc->get_child(2);
    auto c21 = m_data->hc->get_child(1);
    auto c22 = m_data->hc->get_child(3);
    size_t dim = m_data->hc->get_xdim() / 2;
    Matrix* u1 = c11;
    Matrix* u2 = new Matrix(dim, dim);
    Matrix* u3 = new Matrix(dim, dim);
    Matrix* u4 = new Matrix(dim, dim);
    Matrix* u5 = c12;
    Matrix* u6 = c21;
    Matrix* u7 = c22;

    // int size = subproblems.size();
    auto task_u2_ha = ((StrassenData_t*)((Strassen*)subproblems[4]->data))->hc;
    auto task_u2_hb = ((StrassenData_t*)((Strassen*)subproblems[2]->data))->hc;    
    auto task_u3_hb = ((StrassenData_t*)((Strassen*)subproblems[0]->data))->hc;
    auto task_u4_hb = ((StrassenData_t*)((Strassen*)subproblems[1]->data))->hc;
    auto task_u5_hb = ((StrassenData_t*)((Strassen*)subproblems[3]->data))->hc;
    auto task_u6_hb = ((StrassenData_t*)((Strassen*)subproblems[6]->data))->hc;
    auto task_u1_hb = ((StrassenData_t*)((Strassen*)subproblems[5]->data))->hc;

    Strassen *task_u2 = new Strassen(new StrassenData_t(task_u2_ha, task_u2_hb, u2), cpu_add, gpu_add, this);
    Strassen *task_u3 = new Strassen(new StrassenData_t(u2, task_u3_hb, u3), cpu_add, gpu_add, this);
    Strassen *task_u4 = new Strassen(new StrassenData_t(u2, task_u4_hb, u4), cpu_add, gpu_add, this);
    Strassen *task_u7 = new Strassen(new StrassenData_t(u3, task_u4_hb, u7), cpu_add, gpu_add, this);
    Strassen *task_u5 = new Strassen(new StrassenData_t(u4, task_u5_hb, u5), cpu_add, gpu_add, this);
    Strassen *task_u6 = new Strassen(new StrassenData_t(u3, task_u6_hb, u6), cpu_sub, gpu_sub, this);
    Strassen *task_u1 = new Strassen(new StrassenData_t(task_u2_ha, task_u1_hb, u1), cpu_add, gpu_add, this);
    Task* mer_task = new Task({task_u2, task_u3, task_u4, task_u5, task_u6, task_u7, task_u1});
    mer_task->run(this, 'g');

    subproblems.push_back(new Strassen(new StrassenData_t(u2, u3, u4), nullptr, nullptr, nullptr));

    return ;
}

void StrassenProblem::mergePostPro(std::vector<Problem *> subproblems) {
    auto m_data = (StrassenData_t*)data;
    int size = subproblems.size();
    size_t dim = m_data->hc->get_xdim() / 2;
// #pragma unroll 8
    for(int i = 0; i < size; ++i){
        auto cur_data = (StrassenData_t*)((Strassen*)subproblems[i]->data);
        if(cur_data->ha->get_ld() == dim){
            delete cur_data->ha;
        }

        if(cur_data->hb->get_ld() == dim){
            delete cur_data->hb;
        }

        if(cur_data->hc->get_ld() == dim){
            delete cur_data->hc;
        }
    }
}

/***************************
* cpu multiplication operation
***************************/
void cpu_mul(Basedata_t* d) {
    auto data = (StrassenData_t*) d;
    int dim = data->ha->get_xdim();
    _TYPE* ha = data->ha->get_cdata();
    _TYPE* hb = data->hb->get_cdata();
    _TYPE* hc = data->hc->get_cdata();
    int lda = data->ha->get_ld();
    int ldb = data->hb->get_ld();
    int ldc = data->hc->get_ld();

    CPU_GEMM(CblasColMajor, CblasNoTrans, CblasNoTrans,
        dim, dim, dim,
        1, ha, lda, hb,
        ldb, 0, hc, ldc);
}

/***************************
* gpu multiplication operation
***************************/
void gpu_mul(Basedata_t* d) {
    auto data = (StrassenData_t*) d;
    
    int dim = data->ha->get_xdim();
    _TYPE* ha = data->ha->get_gdata();
    _TYPE* hb = data->hb->get_gdata();
    _TYPE* hc = data->hc->get_gdata();
    int lda = data->ha->get_ld();
    int ldb = data->hb->get_ld();
    int ldc = data->hc->get_ld();
    // auto cublas_handle = handle();
    const double p_one = 1;    
    const double zero = 0;
    // GPU_GEMM(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, dim,
    //     dim, dim, &p_one, ha,lda, hb, ldb, &zero,
    //     hc, ldc);
    gemm(data->ha->get_gdata(),data->hb->get_gdata(), data->hc->get_gdata(),
        data->ha->get_xdim(), data->ha->get_ld(), data->hb->get_ld(), data->hc->get_ld(), stream());

    // std::cout << "mul....." << std::endl;
    // data->hc->copy_from(data->hc->get_cdata(), data->hc->get_gdata(), Runtime::get_instance().get_cpu());
    // // // cudaMemcpy2DAsync(data->hc->get_cdata(), data->hc->get_ld()*sizeof(_TYPE), data->hc->get_gdata(),
    // // //     data->hc->get_ld()*sizeof(_TYPE), dim*sizeof(_TYPE), dim, cudaMemcpyDeviceToHost, stream()
    // // //     );
    // _TYPE* cdata = data->hc->get_cdata();
    // for(int i = 0; i < dim*dim; ++i){
	// 	//std::cout << i << ":" << cdata[i] << std::endl;
	// 	printf("(i:%d, value:%f)", i, cdata[i]);
	// 	if(i > 0 && i % dim == 0) printf("\n");
	// }
    // std::cout <<"--------------------"<<std::endl;
}

/***************************
* cpu addition operation
***************************/
void cpu_add(Basedata_t* d) {
    auto data = (StrassenData_t*) d;
    const double p_one = 1;
    CPU_GEAM('C', 'N', 'N', data->ha->get_xdim(),
            data->ha->get_ydim(), p_one, data->ha->get_cdata(), data->ha->get_ld(),
            p_one, data->hb->get_cdata(), data->hb->get_ld(), data->hc->get_cdata(),
            data->hc->get_ld());
    
}

/***************************
* gpu addition operation
***************************/
void gpu_add(Basedata_t* d) {
    auto data = (StrassenData_t*) d;
    // auto cublas_handle = handle();
    const double p_one = 1;
    // GPU_GEAM(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, data->ha->get_xdim(),
    //          data->ha->get_ydim(), &p_one, data->ha->get_gdata(), data->ha->get_ld(),
    //          &p_one, data->hb->get_gdata(), data->hb->get_ld(), data->hc->get_gdata(),
    //          data->hc->get_ld());
    sumMatrix(data->ha->get_gdata(),data->hb->get_gdata(), data->hc->get_gdata(),
        data->ha->get_xdim(), data->ha->get_ld(), data->hb->get_ld(), data->hc->get_ld(), stream());

    // std::cout << "add....." << std::endl;
    // data->hc->copy_from(data->hc->get_cdata(), data->hc->get_gdata(), Runtime::get_instance().get_cpu());
    // auto dim = data->ha->get_xdim();
    // // cudaMemcpy2DAsync(data->hc->get_cdata(), data->hc->get_ld()*sizeof(_TYPE), data->hc->get_gdata(),
    // //     data->hc->get_ld()*sizeof(_TYPE), dim*sizeof(_TYPE), dim, cudaMemcpyDeviceToHost, stream()
    // //     );
    // _TYPE* cdata = data->hc->get_cdata();
    // for(int i = 0; i < dim*dim; ++i){
	// 	//std::cout << i << ":" << cdata[i] << std::endl;
	// 	printf("(i:%d, value:%f)", i, cdata[i]);
	// 	if(i > 0 && i % dim == 0) printf("\n");
	// }
    // std::cout <<"--------------------"<<std::endl;
}

/***************************
* cpu sub operation
***************************/
void cpu_sub(Basedata_t* d) {
    auto data = (StrassenData_t*) d;
    const double p_one = 1;
    int n_one = -1;
    CPU_GEAM('C', 'N', 'N', data->ha->get_xdim(),
            data->ha->get_ydim(), p_one, data->ha->get_cdata(), data->ha->get_ld(),
            n_one, data->hb->get_cdata(), data->hb->get_ld(), data->hc->get_cdata(),
            data->hc->get_ld());
    
}

/***************************
* gpu sub operation
***************************/
void gpu_sub(Basedata_t* d) {
    auto data = (StrassenData_t*) d;
    const double p_one = 1;
    const double n_one = -1;
    // auto cublas_handle = handle();
    // GPU_GEAM(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, data->ha->get_xdim(),
    //          data->ha->get_ydim(), &p_one, data->ha->get_gdata(), data->ha->get_ld(),
    //          &n_one, data->hb->get_gdata(), data->hb->get_ld(), data->hc->get_gdata(),
    //          data->hc->get_ld());

    subMatrix(data->ha->get_gdata(),data->hb->get_gdata(), data->hc->get_gdata(),
        data->ha->get_xdim(), data->ha->get_ld(), data->hb->get_ld(), data->hc->get_ld(), stream());
    // std::cout << "sub....." << std::endl;
    // auto dim = data->ha->get_xdim();
    // // cudaMemcpy2DAsync(data->hc->get_cdata(), data->hc->get_ld()*sizeof(_TYPE), data->hc->get_gdata(),
    // //     data->hc->get_ld()*sizeof(_TYPE), dim*sizeof(_TYPE), dim, cudaMemcpyDeviceToHost, stream()
    // //     );
    // data->hc->copy_from(data->hc->get_cdata(), data->hc->get_gdata(), Runtime::get_instance().get_cpu());
    // _TYPE* cdata = data->hc->get_cdata();
    // for(int i = 0; i < dim*dim; ++i){
	// 	//std::cout << i << ":" << cdata[i] << std::endl;
	// 	printf("(i:%d, value:%f)", i, cdata[i]);
	// 	if(i > 0 && i % dim == 0) printf("\n");
	// }
    // std::cout <<"--------------------"<<std::endl;
    
}