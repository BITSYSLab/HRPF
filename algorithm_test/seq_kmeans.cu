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
#include <omp.h>
#include <sys/time.h>
using namespace std;
static int cols = 8;
static int rows;
#define k 3

class KMEANS
{
private:
    double *dataSet; // rows*cols

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
    KMEANS(double *train, double *index, double *dist, double *centr[k])
    {
        dataSet = train;
        perMinDist = dist;
        perIndex = index;
        // centroids = centr;
        for (int i = 0; i < k; ++i)
        {
            centroids[i] = centr[i];
        }
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
    int clusterChanged = 10;

    struct timeval start, end;
    gettimeofday(&start, NULL);
    while (clusterChanged)
    {
        clusterChanged--; //= false;
#pragma omp parallel for num_threads(22)
        for (int i = 0; i < rows; ++i)
        {
            int minIndex = -1;
            double minDist = INT_MAX;
            for (int j = 0; j < k; ++j)
            {
                double sum = 0;
                for (int c = 0; c < cols; ++c)
                {
                    sum += sqrt((centroids[j][c] - dataSet[i + c * rows]) * (centroids[j][c] - dataSet[i + c * rows]));
                }
                if (sum < minDist)
                {
                    minDist = sum;
                    minIndex = j;
                }
            }

            perIndex[i] = minIndex;
            perMinDist[i] = minDist;
            // if(clusterAssment[i].minIndex != minIndex) {
            //     // clusterChanged = true;
            //     clusterAssment[i].minIndex = minIndex;
            //     clusterAssment[i].minDist = minDist;
            // }
        }

        // #pragma omp parallel for num_threads(16)
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
                        vec[j] += dataSet[i + j * rows];
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
    }
    gettimeofday(&end, NULL);
    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    std::cout << seconds << std::endl;
    std::ofstream ofs;                     //定义流对象
    ofs.open("km_seq_6144.txt", ios::out); //以写的方式打开文件
    for (int i = 0; i < rows; ++i)
    {
        ofs << clusterAssment[i].minIndex << std::endl;
    }
    ofs.close();

    delete[] dataSet;
    delete[] perIndex;
    delete[] perMinDist;
    for (int i = 0; i < k; ++i)
    {
        delete centroids[i];
    }
}

void KMEANS::setCentroids(tMinMax &tminmax, int idx)
{
    double rangeIdx = tminmax.Max - tminmax.Min;
    // for(int i = 0; i < k; ++i)
    // {
    //     // std::cout << "k:" << i << std::endl;
    //     double tmp = rangeIdx * (rand() / (double)RAND_MAX);
    //     std::cout <<"tmp:"<< tmp <<std::endl;
    //     centroids[i][idx] = tminmax.Min + tmp;
    // }
    if (idx == 0)
    {
        centroids[0][idx] = tminmax.Min + 0.261151;
        centroids[1][idx] = tminmax.Min + 0.334051;
        centroids[2][idx] = tminmax.Min + 0.317798;
    }
    else
    {
        centroids[0][idx] = tminmax.Min + 0.914228;
        centroids[1][idx] = tminmax.Min + 0.604528;
        centroids[2][idx] = tminmax.Min + 0.578463;
    }
}

typename KMEANS::tMinMax KMEANS::getMinMax(int idx)
{
    double min, max;
    dataSet[0 + idx * rows] > dataSet[1 + idx * rows] ? (max = dataSet[0 + idx * rows], min = dataSet[1 + idx * rows]) : (max = dataSet[1 + idx * rows], min = dataSet[0 + idx * rows]);

    for (int i = 2; i < rows; ++i)
    {
        if (dataSet[i + idx * rows] < min)
            min = dataSet[i + idx * rows];
        else if (dataSet[i + idx * rows] < max)
            max = dataSet[i + idx * rows];
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
    fin.open("kmdata_6144_8.txt");

    if (!fin)
    {
        std::cout << "can not open the file data.txt" << std::endl;
        exit(1);
    }

    /* input the dataSet */
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            fin >> dataSet[i + j * rows];
        }
    }
    fin.close();
}

int main(int argc, char **argv)
{
    rows = std::atoi(argv[1]);
    double *train = new double[rows * cols];
    double *pIndex = new double[rows];
    double *pMinDist = new double[rows];
    double *centr[k];
    for (int i = 0; i < k; ++i)
    {
        centr[i] = new double[cols];
    }
    KMEANS kms(train, pIndex, pMinDist, centr);
    // std::cout << "kms construct..." << std::endl;
    kms.loadDataSet();
    // std::cout << "load dataset succ" << std::endl;
    kms.randCent();
    // std::cout << "rand centroid" << std::endl;
    kms.kmeans();
    return 0;
}
