#include<iostream>
#include<map>
#include<vector>
#include<stdio.h>
#include<cmath>
#include<cstdlib>
#include<algorithm>
#include<fstream>
#include <thrust/sort.h>

#include <omp.h>
#include <sys/time.h>

static int cols = 8;
static int rows;
static int num_test;
static int test_label;
// static int k = 5;

class KNN {
private:
    double *dataSet;
    int *label;
    double *test;
    double *map_index_dist;
    // double *map_label_freq;
    int k;
    double get_distance(double *d1, double *d2, int index);

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
}

double KNN:: get_distance(double *d1,double *d2, int index)
{
	double sum = 0;
	for(int i=0;i<cols;i++)
	{
		sum += sqrt((d1[i*rows + index]-d2[i])*(d1[i*rows + index]-d2[i]));
	}
 
//	cout<<"the sum is = "<<sum<<endl;
	return sum;
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
    /******************************origin imple*******************************/
	double distance;
	int i;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    #pragma omp parallel for num_threads(8)
	for(i=0;i<rows;i++)
	{
        double sum = 0;
        for(int j=0;j<cols;j++)
        {
            sum += sqrt((dataSet[j*rows + i]-test[j])*(dataSet[j*rows + i]-test[j]));
        }
		// distance = get_distance(dataSet,test, i);
		//<key,value> => <i,distance>
		map_index_dist[i] = sum;
        // std::cout << sum << std::endl;
	}
    gettimeofday(&end, NULL);
    double seconds = (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
    // std::cout << seconds << std::endl;
	// //traverse the map to print the index and distance
	// map<int,double>::const_iterator it = map_index_dis.begin();
	// while(it!=map_index_dis.end())
	// {
	// 	cout<<"index = "<<it->first<<" distance = "<<it->second<<endl;
	// 	it++;
	// }
    
	int *index = new int[rows];
	#pragma omp parallel for
	for(int i = 0; i < rows; ++i){
		index[i] = i;
	}

	// thrust::sort_by_key(index, index+rows, map_index_dist);
    sortByKey(index, rows, map_index_dist);
	std::map<int, int> m;
    for(int i = 0; i < k; ++i) {
        // std::cout << map_index_dist[index[i]] << label[index[i]] << std::endl;
        m[label[index[i]]]++;
    }

    int t_label = -1;
    int freq = 0;
    for(auto it = m.begin(); it != m.end(); ++it){
        // std::cout << it->second << " " << it->first << std::endl;
        if(it->second > freq){
            freq = it->second;
            t_label = it->first;
        }
    }

    std::cout << seconds << " " << t_label  << " " << test_label<< std::endl;
    delete []dataSet;
    delete []test;
    delete []map_index_dist;
    delete []index;
    delete []label;
}

int main(int argc, char **argv) {
    rows = 4000;
    num_test = std::atoi(argv[1]);
    double* train = new double[rows*cols];
    double* test = new double[cols];
    int*   label = new int[rows];
    double* index_dist = new double[rows];
    KNN knn(3, train, test, label, index_dist);
    
    knn.get_all_distance();
    
    return 0;
}
