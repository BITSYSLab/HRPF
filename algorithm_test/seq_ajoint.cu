/*
 * @Author: your name
 * @Date: 2022-03-03 10:09:18
 * @LastEditTime: 2022-03-12 14:48:19
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \git_file_graduate\HRPA_12\algorithm\parallel_for\parallel_for_harness.cpp
 */
#pragma once
#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <fstream>

static int length;
#define PI 3.14159

// void cfor_func(double* ar, double* ai, double* tr, double* ti){
//     size_t s_i = 0;
//     size_t e_i = length;
//     size_t s_j = 0;
//     size_t e_j = length;
    
//     #pragma omp parallel for num_threads(16)
//     for(int i = s_i; i < e_i; ++i){
//         tr[i] = 0; double wnr = 0;
//         ti[i] = 0; double wni = 0;
//         for(int j = s_j; j < e_j; ++j) {
//            wnr = cos(2.0 * PI * i * j / length );
//            wni = sin(2.0 * PI * i * j / length);
//            tr[i] += (ar[j] * wnr - ai[j] * wni);
//            ti[i] += (ar[j] * wni + ai[j] * wnr);
//         } 
//     }
// }

void seq_cfor_func(double* a, double* b, double* c){
    // std::cout << "enter seq ..." << std::endl;
    size_t s_i = 0;
    size_t e_i = length*length;
    size_t s_j = 0;
    size_t e_j = length*length;
    // std::cout << s_i << s_j << e_i << e_j << std::endl;
    // #pragma omp parallel for simd
    for(int i = s_i; i < e_i; ++i){
        double cur_c = 0;
        for(int j = i; j < e_j; ++j) {
           cur_c += 5.5*b[j]*a[j-i];
        } 
        c[i] = cur_c;
    }
}

void initialize(double* datar, double* datai, int length) {
    // srand48(time(NULL));
    // for(int i = 0; i < length; ++i){
    //     data[i] = (double)(rand() % 100);
    // }
    std::ifstream fin;
    fin.open("datadft.txt");
 
	if(!fin)
	{
		std::cout<<"can not open the file data.txt"<<std::endl;
		exit(1);
	}

    for(int i = 0; i < length; ++i){
        fin >> datar[i] >> datai[i];
    }
}

void print(double* datar, int length) {
    for(int i = 0; i < length; ++i){
        std::cout << datar[i] << " " ;;//<< datai[i] << std::endl;
        if(i && i % 4 == 0) std::cout << std::endl;
    }
}

int main(int argc, char **argv){
    std::size_t N = std::atoi(argv[1]);
    length = N;
    // std::cout << "length:" << length << std::endl;
    double* a = new double[length*length];
    double* b = new double[length*length];
    double* c = new double[length*length];
    
    initialize(a, b, length*length);
    struct timeval start, end;

    gettimeofday(&start, NULL);
    seq_cfor_func(a, b, c);
    gettimeofday(&end, NULL);
    double seconds =  (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    std::cout << seconds << std::endl;
    print(c, length*length);
   
    delete []a;
    delete []b;
    delete []c;
 
    return 0;
}
