/*
 * @Author: your name
 * @Date: 2021-11-30 09:07:33
 * @LastEditTime: 2022-02-03 18:51:54
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_NEW\algorithm\strassen_harness.cpp
 */
#pragma once

#include <stdio.h>
#include <sys/time.h>
//#include <mkl.h>

#include <string>
#include <iostream>
#include "framework/framework.h"
#include "strassenProblem.h"
#include "tool/initializer.h"

int main(int argc, char **argv) {
    mkl_set_num_threads(16);
    
    std::size_t dim = std::atoi(argv[1]);
    std::string interleaving = argv[2];
	Framework::init();
    Matrix* ha = new Matrix(dim, dim);
    Matrix* hb = new Matrix(dim, dim);
    Matrix* hc = new Matrix(dim, dim);
    // std::cout << "init start" << std::endl;
   	
    initialize(dim, ha, hb);
   	// std::cout<<"init end" << std::endl;
   	// StrassenData_t *data = new StrassenData_t(ha, hb, hc);
    StrassenProblem* problem = new Strassen(new StrassenData_t(ha, hb, hc), cpu_mul, gpu_mul, nullptr);
   	// std::cout << "problem construct end..." << std::endl; 
	//Framework::init();
	struct timeval start, end;
    gettimeofday(&start, NULL);
	//Framework::init();
	Framework::solve(problem, interleaving);
	// hc->access(Runtime::get_instance().get_cpu(), MemAccess::R);
    gettimeofday(&end, NULL);
    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    // 
	hc->access(Runtime::get_instance().get_cpu(), MemAccess::R);
	_TYPE* cdata = hc->get_cdata();
	// _TYPE* gdata = hc->get_gdata();
	// hc->copy_from(cdata, gdata, Runtime::get_instance().get_cpu());
	
	// for(int i = 0; i < dim*dim; ++i){
	// 	//std::cout << i << ":" << cdata[i] << std::endl;
	// 	printf("(i:%d, value:%f)", i, cdata[i]);
	// 	if(i > 0 && i % dim == 0) printf("\n");
	// }
	// std::cout << "-------------------"<< std::endl;
	// _TYPE* gdata = hc->get_gdata();
	// hc->copy_from(cdata, gdata, Runtime::get_instance().get_cpu());
	
	// for(int i = 0; i < dim*dim; ++i){
	// 	//std::cout << i << ":" << cdata[i] << std::endl;
	// 	printf("(i:%d, value:%f)", i, cdata[i]);
	// 	if(i > 0 && i % dim == 0) printf("\n");
	// }
	std::cout << seconds << std::endl; 
    delete problem;
    delete ha;
    delete hb;
    delete hc;
	// std::cout << "pro end...exit" << std::endl;
	return 0;
}
