#include<iostream>
#include<map>
#include<vector>
#include<stdio.h>
#include<cmath>
#include<cstdlib>
#include<algorithm>
#include<fstream>
#include <thrust/sort.h>
#include "datastructture/matrix.h"
#include "datastructture/arraylist.h"
#include "algorithm/parallel_for_zero/parallel_for_zb.h"
#include "framework/framework.h"
#include "tool/helper.h"
// #include "algorithm/parallel_for/parallel_for_harness.hpp"
#include <omp.h>
#include <sys/time.h>

static int cols = 8;
static int rows;
static int num_test;
static int test_label;
// static int k = 5;

struct UserData_t : public Basedata_t{
public:
    UserData_t(std::vector<Matrix*>m_bf, std::vector<ArrayList*> buf
        ) : m_buffer(m_bf), v_buffer(buf){
        }

public:
    std::vector<Matrix*> m_buffer;
    std::vector<ArrayList*> v_buffer;
};

void cfor_func(Basedata_t* data){
    auto d = (loopData_t*)data;
    auto a = ((UserData_t*)(d->buffer))->m_buffer[0]->get_cdata();
    auto b = ((UserData_t*)(d->buffer))->v_buffer[0]->get_cdata();
    auto c = ((UserData_t*)(d->buffer))->v_buffer[1]->get_cdata();
    
    size_t lda = ((UserData_t*)(d->buffer))->m_buffer[0]->get_ld();
    
    size_t s_i = d->start;
    size_t e_i = d->end;
    size_t s_j = 0;
    size_t e_j = cols;
    // std::cout << s_i << s_j << e_i << e_j << std::endl;
    #pragma omp parallel for 
    for(int i = s_i; i < e_i; ++i){
        double loc = 0;
        for(int j = s_j; j < e_j; ++j) {
            loc += sqrt((a[i + j * lda] - b[j])*(a[i + j * lda] - b[j]));
            // std::cout << a[i + j * lda] << std::endl;
        }
        c[i] = loc;
    }
}

__global__ void kernel_2DKNN(size_t s_i, size_t e_i, size_t s_j, size_t e_j,
    size_t lda, size_t ldb, size_t ldc,
    size_t chunk, double* a, double* b, double* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int start_i = s_i + tid * chunk;
    int end_i = start_i + chunk < e_i ? start_i + chunk : e_i;

    for(int i = start_i; i < end_i; ++i){
        double loc = 0.0;
        for(int j = s_j; j < e_j; ++j) {
            loc += sqrt((a[i + j * lda] - b[j])*(a[i + j * lda] - b[j]));
        }
        c[i] = loc;
    }  
}

void gfor_func(Basedata_t* data){
    auto d = (loopData_t*)data;
    auto a = ((UserData_t*)(d->buffer))->m_buffer[0]->get_gdata();
    auto b = ((UserData_t*)(d->buffer))->v_buffer[0]->get_gdata();
    auto c = ((UserData_t*)(d->buffer))->v_buffer[1]->get_gdata();
    
    size_t lda = ((UserData_t*)(d->buffer))->m_buffer[0]->get_ld();
    
    size_t s_i = d->start;
    size_t e_i = d->end;
    size_t s_j = 0;
    size_t e_j = cols;
    
    int blocks_required = 1;
    int threads_per_block = 1024;
    int chunk_size = 1;
    int size = e_i - s_i;
    if(size % (threads_per_block * chunk_size)) {
        blocks_required = size / (threads_per_block * chunk_size) + 1;
    }
    else {
        blocks_required = size / (threads_per_block * chunk_size);
    }
    cudaStream_t stream_ = stream();
    kernel_2DKNN<<<blocks_required, threads_per_block, 0, stream_>>>(s_i, e_i, s_j, e_j, lda, 0, 0, 
        chunk_size, a, b, c);
}

class KNN {
private:
    double *dataSet;
    int *label;
    double *test;
    double *map_index_dist;
    Matrix* trainData;
    ArrayList* testData;
    ArrayList* allDistance;
    // double *map_label_freq;
    int k;

public:
    KNN(int k, double* train, double* test, int* label, 
        double* index_dist);
    void get_all_distance();
    void get_max_freq_lable();
};

KNN::KNN(int k, double* train, double* test, int* label, 
        double* index_dist)
{
    
    std::ifstream fin, tfin;
    std::ifstream finl, tfinl;
    // ofstream fout;
	this->k = k;
    dataSet = train; this->test = test; this->label = label;
    map_index_dist = index_dist; 
	fin.open("train.txt");
    finl.open("labeltrain.txt");
	if(!fin)
	{
		std::cout<<"can not open the file data.txt"<<std::endl;
		exit(1);
	}
 
    /* input the dataSet */ 
	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			fin>>dataSet[i + j*rows];
		}
		finl>>label[i];
	}
    
    tfin.open("test.txt");
    tfinl.open("labeltest.txt");
	// std::cout<<"please input the test data :"<<std::endl;
	/* inuput the test data */
    for(int j = 0; j < num_test; ++j)
	{
        for(int i=0;i<cols;i++)
		    tfin>>test[i];
        tfinl >> test_label;
    }    
    
    trainData = new Matrix(dataSet, rows, cols);
    testData = new ArrayList(test, cols);
    allDistance = new ArrayList(map_index_dist, rows);
}

void sortByKey(int* data, int len, double* key){
    for(int i = 0; i < len-1; ++i){
        for(int j = 0; j < len - 1-i; ++j){
            if(key[j] > key[j+1]){
                int td = data[j];
                data[j] = data[j+1];
                data[j+1] = td;

                double tk = key[j];
                key[j] = key[j+1];
                key[j+1] = tk;
            }
        }
    }
}
/*
 * calculate all the distance between test data and each training data
 */
void KNN:: get_all_distance()
{
    /*****************************parallel**********************************/
    UserData_t* user = new UserData_t({trainData},{testData, allDistance});
    struct timeval start, end;
    gettimeofday(&start, NULL);
    parallel_for(new loopData_t(0, rows, user), cfor_func, gfor_func);
	
	// allDistance->access(Runtime::get_instance().get_cpu(), MemAccess::R);
    gettimeofday(&end, NULL);
    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    // std::cout << seconds << std::endl;
    int *index = new int[rows];
	#pragma omp parallel for
	for(int i = 0; i < rows; ++i){
		index[i] = i;
	}

	// thrust::sort_by_key(index, index+rows, map_index_dist);
	sortByKey(index, rows, map_index_dist);
    std::map<int, int> m;
    for(int i = 0; i < k; ++i) {
        m[label[index[i]]]++;
    }

    int t_label = -1;
    int freq = 0;
    for(auto it = m.begin(); it != m.end(); ++it){
        if(it->second > freq){
            freq = it->second;
            t_label = it->first;
        }
    }

    // std::cout << "The test data belongs to the " << t_label  << " true:" << test_label << std::endl;
    std::cout << seconds << " " << t_label  << " " << test_label<< std::endl;
    delete trainData;
    delete testData;
    delete allDistance;
    delete user;
    delete []index;
    delete []label;
}

int main(int argc, char **argv) {
    Framework::init();
    num_test = std::atoi(argv[1]);
    rows = 4000;
    double* train;
    cudaHostAlloc(&train, rows*cols * sizeof(double), cudaHostAllocMapped);
    double* test;
    cudaHostAlloc(&test, cols * sizeof(double), cudaHostAllocMapped);
    int*   label = new int[rows];
    double* index_dist;
    cudaHostAlloc(&index_dist, rows * sizeof(double), cudaHostAllocMapped);
    KNN knn(5, train, test, label, index_dist);
    
    knn.get_all_distance();
   
    return 0;
}
