/*
 * @Author: your name
 * @Date: 2022-03-02 19:50:13
 * @LastEditTime: 2022-03-03 14:27:13
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_12\algorithm\parallel_for\parallel_for_problem.h
 */
#pragma once
#include "framework/problem.h"
#include "framework/task.h"
#include "common/gpu_device.h"
#include "datastructture/arraylist.h"
#include <string>
#include <initializer_list>
#include <bitset>

struct loopData_t : public Basedata_t{
public:
    loopData_t(size_t s, size_t e, Basedata_t* buf
        ) : buffer(buf){
            start = s;
            end = e;
        }

public:
    Basedata_t* buffer;
    size_t start;
    size_t end;
};

class CplusLoopInc : public Problem{
public:
    std::vector<Problem*> split() override;
    void merge(std::vector<Problem*>& subproblems) override;
    bool mustRunBaseCase();
    bool canRunBaseCase(int index);
public:
    CplusLoopInc(Basedata_t* m_data, Function _cf, Function _gf, Problem* par) {
        data = m_data;
        cpu_func = _cf;
        gpu_func = _gf;
        parent = par;
        m_mask = std::bitset<T_SIZE>("10000");
    }
    virtual ~CplusLoopInc() {
        if(data != nullptr){
            delete data;
            data = nullptr;
        }
    }

    void Input()  {}
    void Output()  {}
    void IO(Basedata_t* m_data) {
        
    }

};

void parallelForI2D(Basedata_t* data, Function cf, Function gf);
