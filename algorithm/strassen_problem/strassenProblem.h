/*
 * @Author: your name
 * @Date: 2021-11-24 14:41:36
 * @LastEditTime: 2022-02-13 21:16:22
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_NEW\algorithm\strassen_problem\strassenProblem.h
 */
#pragma once

#include <mkl.h>
#include "datastructture/matrix.h"
#include "framework/problem.h"
#include "framework/task.h"
#include "common/gpu_device.h"
#include <memory>

struct StrassenData_t:public Basedata_t
{
    /* data */
public:
    Matrix* ha;
    Matrix* hb;
    Matrix* hc;

public:
    StrassenData_t(Matrix* data_a, Matrix* data_b, Matrix* data_c){
        ha = data_a;
        hb = data_b;
        hc = data_c;
    }
};

class StrassenProblem: public Problem {
public:
    //typedef typename std::function<void(Device*, MatrixSt*, MatrixSt*, MatrixSt*)> Function;
    // std::vector<Task*> split() override;
    std::vector<Problem*> split();
    void merge(std::vector<Problem*>& subproblems) override;
    void mergePostPro(std::vector<Problem *> subproblems);
    bool mustRunBaseCase();
    bool canRunBaseCase(int index);
public:
    StrassenProblem(Basedata_t* d, Function cf, Function gf, Problem* par); 
    ~StrassenProblem(){
        if(data != nullptr){
            delete data;
            data = nullptr;
        }
    }

    void Input() override;
    void Output() override;
    void IO(Basedata_t* m_data) override;
};

typedef StrassenProblem Strassen;

#define GPU_GEMM cublasDgemm
#define GPU_GEAM cublasDgeam
#define CPU_GEMM cblas_dgemm
#define CPU_GEAM mkl_domatadd

/******************function declare**********************/
void cpu_mul(Basedata_t*);
void gpu_mul(Basedata_t*);
void cpu_add(Basedata_t*);
void gpu_add(Basedata_t*);
void cpu_sub(Basedata_t*);
void gpu_sub(Basedata_t*);