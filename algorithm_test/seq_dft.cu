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

void cfor_func(double* ar, double* ai, double* tr, double* ti){
    size_t s_i = 0;
    size_t e_i = length;
    size_t s_j = 0;
    size_t e_j = length;
    
    #pragma omp parallel for num_threads(16)
    for(int i = s_i; i < e_i; ++i){
        tr[i] = 0; double wnr = 0;
        ti[i] = 0; double wni = 0;
        for(int j = s_j; j < e_j; ++j) {
           wnr = cos(2.0 * PI * i * j / length );
           wni = sin(2.0 * PI * i * j / length);
           tr[i] += (ar[j] * wnr - ai[j] * wni);
           ti[i] += (ar[j] * wni + ai[j] * wnr);
        } 
    }
}

void seq_cfor_func(double* ar, double* ai, double* tr, double* ti){
    // std::cout << "enter seq ..." << std::endl;
    size_t s_i = 0;
    size_t e_i = length;
    size_t s_j = 0;
    size_t e_j = length;
    // std::cout << s_i << s_j << e_i << e_j << std::endl;
    //#pragma omp parallel for simd// private(j,i)
    for(int i = s_i; i < e_i; ++i){
        tr[i] = 0; double wnr = 0;
        ti[i] = 0; double wni = 0;
        for(int j = s_j; j < e_j; ++j) {
           wnr = cos(2.0 * PI * i * j / length);
           wni = sin(2.0 * PI * i * j / length);

        //    std::cout << i << j << wnr << " " << wni << std::endl;
           tr[i] += (ar[j] * wnr - ai[j] * wni);
           ti[i] += (ar[j] * wni + ai[j] * wnr);
        } 
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

void print(double* datar, double* datai, int length) {
    for(int i = 0; i < length; ++i){
        std::cout << datar[i] << " " ;;//<< datai[i] << std::endl;
        if(i && i % 4 == 0) std::cout << std::endl;
    }
}

int main(int argc, char **argv){
    std::size_t N = std::atoi(argv[1]);
    length = N;
    // std::cout << "length:" << length << std::endl;
    double* datar = new double[length];
    double* datai = new double[length];
    double* tempr = new double[length];
    double* tempi = new double[length];
    initialize(datar, datai, length);
    // initialize(datai, length);

    struct timeval start, end;
    // gettimeofday(&start, NULL);
    // cfor_func(datar, datai, tempr, tempi);
    // gettimeofday(&end, NULL);
    // double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    // std::cout << seconds << std::endl;
    // print(tempr, tempi, length);

    gettimeofday(&start, NULL);
    seq_cfor_func(datar, datai, tempr, tempi);
    gettimeofday(&end, NULL);
    double seconds =  (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    std::cout << seconds << std::endl;
    print(tempr, tempi, length);
    // auto da = data3->get_cdata();
    // for(int i = 0; i < length; ++i){
    //     for(int j = 0; j < length; ++j){
    //         std::cout << da[i + length*j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    delete []datar;
    delete []datai;
    delete []tempr;
    delete []tempi;
    return 0;
}
