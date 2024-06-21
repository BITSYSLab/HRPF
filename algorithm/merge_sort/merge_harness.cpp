/*
 * @Author: your name
 * @Date: 2021-12-03 08:48:41
 * @LastEditTime: 2022-02-11 09:05:47
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_NEW\algorithm\merge_sort\merge_harness.cpp
 */
#pragma once

#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <iostream>
#include <fstream>
#include "framework/framework.h"
#include "mergeSortProblem.h"
#include "tool/initializer.h"

void loadData(double* datar, int length) { 
    std::ifstream fin;
    fin.open("algorithm/merge_sort/datamer.txt");
 
	if(!fin)
	{
		std::cout<<"can not open the file data.txt"<<std::endl;
		exit(1);
	}

    for(int i = 0; i < length; ++i){
        fin >> datar[i];
    }
}

int main(int argc, char **argv){
    
    std::size_t length = std::atoi(argv[1]);
    std::string interleaving = argv[2];

    ArrayList* data = new ArrayList(length);
    auto& runtime = Runtime::get_instance();
    auto cpu = runtime.get_cpu();
    (data)->access(cpu, MemAccess::W);
    loadData(data->get_cdata(), length);
    // initialize(data, length);
    Framework::init();
    MergesortProblem* problem = new MergesortProblem(new MergeData_t(data), cpu_sort, gpu_sort, nullptr);
	//std::string mask = "10";
	//problem->set_mask(mask);
	// std::cout << "init problem & threads end" << std::endl;    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    Framework::solve(problem, interleaving);
    data->access(Runtime::get_instance().get_cpu(), MemAccess::R);
    gettimeofday(&end, NULL);

    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    std::cout << seconds << std::endl;
   	// data->access(Runtime::get_instance().get_cpu(), MemAccess::R);
   	_TYPE* dd = data->get_cdata();
	
	// for(int i = 0;  i < length; ++i){
	// 	std::cout << dd[i] <<" ";
	// 	if(i&&i % 16 == 0) std::cout << std::endl;
	// }
    
	delete problem;
	//std::cout << "delete problem..." << std::endl;
	delete data;
    return 0;
}

