#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <time.h> //for srand
#include <limits.h>
#include "datastructture/matrix.h"
#include "datastructture/arraylist.h"
#include "algorithm/parallel_for_zero/parallel_for_inc.h"
#include "framework/framework.h"
#include "tool/helper.h"
// #include "algorithm/parallel_for/parallel_for_harness.hpp"
#include <omp.h>
#include <sys/time.h>
using namespace std;
#define cols 8
static int rows;
#define k 3

struct UserData_t : public Basedata_t
{
public:
    UserData_t(std::vector<ArrayList *> buf) : v_buffer(buf)
    {
    }

public:
    std::vector<Matrix *> m_buffer;
    std::vector<ArrayList *> v_buffer;
};

void cfor_func(Basedata_t *data)
{
    // std::cout << "cpu" << std::endl;
    auto d = (loopData_t *)data;
    auto a0 = ((UserData_t *)(d->buffer))->v_buffer[0]->get_cdata();
    auto a1 = ((UserData_t *)(d->buffer))->v_buffer[1]->get_cdata();
    double *cent[3];
    cent[0] = ((UserData_t *)(d->buffer))->v_buffer[2]->get_cdata();
    cent[1] = ((UserData_t *)(d->buffer))->v_buffer[3]->get_cdata();
    cent[2] = ((UserData_t *)(d->buffer))->v_buffer[4]->get_cdata();
    // cent[3] = ((UserData_t*)(d->buffer))->v_buffer[3]->get_cdata();
    auto dist = ((UserData_t *)(d->buffer))->v_buffer[5]->get_cdata();
    auto index = ((UserData_t *)(d->buffer))->v_buffer[6]->get_cdata();

    // size_t lda = ((UserData_t*)(d->buffer))->m_buffer[0]->get_ld();

    size_t s_i = d->start;
    size_t e_i = d->end;
    size_t s_j = 0;
    size_t e_j = cols;
// std::cout << s_i << s_j << e_i << e_j << std::endl;
#pragma omp parallel for num_threads(16)
    for (int i = s_i; i < e_i; ++i)
    {
        int minIdx = -1;
        double minDst = INT_MAX;

        for (int j = 0; j < 3; ++j)
        {
            double sum = 0;
            // for(int c = s_j; c < e_j; ++c){
            sum += sqrt((a0[i] - cent[j][0]) * (a0[i] - cent[j][0]));
            sum += sqrt((a1[i] - cent[j][1]) * (a1[i] - cent[j][1]));
            // }
            if (sum < minDst)
            {
                minDst = sum;
                minIdx = j;
            }
        }
        dist[i] = minDst;
        index[i] = minIdx;
    }
}

__global__ void kernel_2DKMS(size_t s_i, size_t e_i, size_t s_j, size_t e_j,
                             size_t lda, size_t ldb, size_t ldc,
                             size_t chunk, double *a0, double *a1, double *cent0, double *cent1,
                             double *cent2, double *dist, double *index)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double *cent[3] = {cent0, cent1, cent2};
    int start_i = s_i + tid * chunk;
    int end_i = start_i + chunk < e_i ? start_i + chunk : e_i;

    for (int i = start_i; i < end_i; ++i)
    {
        int minIdx = -1;
        double minDst = INT_MAX;

        for (int j = 0; j < 3; ++j)
        {
            double sum = 0;
            // for(int c = s_j; c < e_j; ++c){
            //     sum += sqrt((a[i + c*lda] - cent[j][c])*(a[i + c*lda] - cent[j][c]));
            // }
            sum += sqrt((a0[i] - cent[j][0]) * (a0[i] - cent[j][0]));
            sum += sqrt((a1[i] - cent[j][1]) * (a1[i] - cent[j][1]));
            if (sum < minDst)
            {
                minDst = sum;
                minIdx = j;
            }
        }

        dist[i] = minDst;
        index[i] = minIdx;
    }
}

void gfor_func(Basedata_t *data)
{
    // std::cout << "gpu" << std::endl;
    auto d = (loopData_t *)data;
    auto a0 = ((UserData_t *)(d->buffer))->v_buffer[0]->get_gdata();
    auto a1 = ((UserData_t *)(d->buffer))->v_buffer[1]->get_gdata();
    auto cent0 = ((UserData_t *)(d->buffer))->v_buffer[2]->get_gdata();
    auto cent1 = ((UserData_t *)(d->buffer))->v_buffer[3]->get_gdata();
    auto cent2 = ((UserData_t *)(d->buffer))->v_buffer[4]->get_gdata();
    // auto cent3 = ((UserData_t*)(d->buffer))->v_buffer[3]->get_gdata();
    auto dist = ((UserData_t *)(d->buffer))->v_buffer[5]->get_gdata();
    auto index = ((UserData_t *)(d->buffer))->v_buffer[6]->get_gdata();

    // size_t lda = ((UserData_t*)(d->buffer))->m_buffer[0]->get_ld();

    size_t s_i = d->start;
    size_t e_i = d->end;
    size_t s_j = 0;
    size_t e_j = cols;

    int blocks_required = 1;
    int threads_per_block = 1024;
    int chunk_size = 1;
    int size = e_i - s_i;
    if (size % (threads_per_block * chunk_size))
    {
        blocks_required = size / (threads_per_block * chunk_size) + 1;
    }
    else
    {
        blocks_required = size / (threads_per_block * chunk_size);
    }
    cudaStream_t stream_ = stream();
    kernel_2DKMS<<<blocks_required, threads_per_block, 0, stream_>>>(s_i, e_i, s_j, e_j, 0, 0, 0,
                                                                     chunk_size, a0, a1, cent0, cent1, cent2, dist, index);
}

class KMEANS
{
private:
    double *dataSet[cols]; // rows*cols

    double *centroids[k]; // k*cols
    double *perMinDist;   // rows
    double *perIndex;     // rows
    typedef struct MinMax
    {
        double Min;
        double Max;
        MinMax(double min, double max) : Min(min), Max(max) {}
    } tMinMax;

    typedef struct Node
    {
        int minIndex;
        double minDist;
        Node(int idx, double dist) : minIndex(idx), minDist(dist) {}
    } tNode;
    std::vector<tNode> clusterAssment;

    tMinMax getMinMax(int idx);
    void setCentroids(tMinMax &tminmax, int idx);
    void initClusterAssment();

public:
    // double* train, double* index, double* dist, double* centr[k]
    KMEANS()
    {
        // dataSet = train;
        // perMinDist = dist;
        // perIndex = index;
        // centroids = centr;
        // for(int i = 0; i < k; ++i){
        //     centroids[i] = centr[i];
        // }
    }
    void loadDataSet();
    void kmeans();
    void randCent();
};

void KMEANS::initClusterAssment()
{
    tNode node(-1, -1);
    for (int i = 0; i < rows; ++i)
    {
        clusterAssment.push_back(node);
    }
}

void KMEANS::kmeans()
{
    initClusterAssment();
    // bool clusterChanged = true;
    int clusterChanged = 20;
    // Matrix* trainData = new Matrix(rows, cols);
    ArrayList *trainData0 = new ArrayList(rows);
    dataSet[0] = trainData0->get_cdata();
    ArrayList *trainData1 = new ArrayList(rows);
    dataSet[1] = trainData1->get_cdata();
    ArrayList *pminDist = new ArrayList(rows);
    perMinDist = pminDist->get_cdata();
    ArrayList *pminIndex = new ArrayList(rows);
    perIndex = pminIndex->get_cdata();
    ArrayList *cent[k];
    for (int i = 0; i < k; ++i)
    {
        cent[i] = new ArrayList(cols);
        centroids[i] = cent[i]->get_cdata();
    }
    loadDataSet();
    std::cout << "load dataset succ" << std::endl;
    randCent();
    UserData_t *user = new UserData_t({trainData0, trainData1, cent[0], cent[1], cent[2], pminDist, pminIndex});
    struct timeval start, end;
    gettimeofday(&start, NULL);
    while (clusterChanged)
    {
        clusterChanged--; //= false;

        Framework::init();

        parallelForI2D(new loopData_t(0, rows, user), cfor_func, gfor_func);
// pminDist->access(Runtime::get_instance().get_cpu(), MemAccess::R);
// pminIndex->access(Runtime::get_instance().get_cpu(), MemAccess::R);
#pragma omp parallel for num_threads(16)
        for (int i = 0; i < rows; ++i)
        {
            if (clusterAssment[i].minIndex != (int)perIndex[i])
            {
                // clusterChanged = true;
                clusterAssment[i].minIndex = (int)perIndex[i];
                clusterAssment[i].minDist = perMinDist[i];
            }
        }

        for (int c = 0; c < k; ++c)
        {
            std::vector<double> vec(cols, 0);
            int cnt = 0;
            for (int i = 0; i < rows; ++i)
            {
                if (clusterAssment[i].minIndex == c)
                {
                    ++cnt;
                    for (int j = 0; j < cols; ++j)
                    {
                        vec[j] += dataSet[j][i];
                    }
                }
            }

            for (int i = 0; i < cols; ++i)
            {
                if (cnt)
                    vec[i] /= cnt;
                centroids[c][i] = vec[i];
            }
        }
        //
    }
    gettimeofday(&end, NULL);
    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    std::cout << seconds << std::endl;
    // UserData_t* user = new UserData_t({trainData},{cent[0], cent[1], cent[2], cent[3], pminDist, pminIndex});
    std::ofstream ofs;            //定义流对象
    ofs.open("km.txt", ios::out); //以写的方式打开文件
    for (int i = 0; i < rows; ++i)
    {
        ofs << clusterAssment[i].minIndex << std::endl;
    }
    ofs.close();

    delete user;
    delete trainData0;
    delete trainData1;
    delete pminDist;
    delete pminIndex;
    for (int i = 0; i < k; ++i)
    {
        delete cent[i];
    }
}

void KMEANS::setCentroids(tMinMax &tminmax, int idx)
{
    double rangeIdx = tminmax.Max - tminmax.Min;
    for (int i = 0; i < k; ++i)
    {
        // std::cout << "k:" << i << std::endl;
        centroids[i][idx] = tminmax.Min + rangeIdx * (rand() / (double)RAND_MAX);
    }
}

typename KMEANS::tMinMax KMEANS::getMinMax(int idx)
{
    double min, max;
    dataSet[idx][0] > dataSet[idx][1] ? (max = dataSet[idx][0], min = dataSet[idx][1]) : (max = dataSet[idx][1], min = dataSet[idx][0]);

    for (int i = 2; i < rows; ++i)
    {
        if (dataSet[idx][i] < min)
            min = dataSet[idx][i];
        else if (dataSet[idx][i] < max)
            max = dataSet[idx][i];
        else
            continue;
    }

    tMinMax tminmax(min, max);
    // std::cout << "finish..." << min << max << std::endl;
    return tminmax;
}

void KMEANS::randCent()
{
    srand(time(NULL));
    for (int j = 0; j < cols; ++j)
    {
        // std::cout << "cols:" << j << std::endl;
        tMinMax tminmax = getMinMax(j);
        setCentroids(tminmax, j);
    }
}

void KMEANS::loadDataSet()
{
    std::ifstream fin;
    fin.open("kmdata.txt");

    if (!fin)
    {
        std::cout << "can not open the file data.txt" << std::endl;
        exit(1);
    }

    /* input the dataSet */
    for (int i = 0; i < rows; i++)
    {
        // for(int j=0;j<cols;j++)
        // {
        fin >> dataSet[0][i] >> dataSet[1][i];
        // }
    }
    fin.close();
}

int main(int argc, char **argv)
{
    rows = std::atoi(argv[1]);
    // double* train;
    // cudaHostAlloc(&train, rows*cols * sizeof(double), cudaHostAllocMapped);
    // double* pIndex;
    // cudaHostAlloc(&pIndex, rows * sizeof(double), cudaHostAllocMapped);
    // double* pMinDist;
    // cudaHostAlloc(&pMinDist, rows * sizeof(double), cudaHostAllocMapped);
    // double* centr[k];
    // for(int i = 0; i < k; ++i){
    //     cudaHostAlloc(&centr[i], cols * sizeof(double), cudaHostAllocMapped);;
    // }train, pIndex, pMinDist, centr
    KMEANS kms;
    // std::cout << "kms construct..." << std::endl;
    // kms.loadDataSet();
    // // std::cout << "load dataset succ" << std::endl;
    // kms.randCent();
    // std::cout << "rand centroid" << std::endl;
    kms.kmeans();
    return 0;
}
