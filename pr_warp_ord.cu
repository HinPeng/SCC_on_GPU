#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iterator>
#include <algorithm>
#include <assert.h>
#include <cstdlib>
//#include <time.h>

using namespace std;

int vertex_num;
int edge_num;
string fn;
float const PAGERANK_COEFFICIENT = 0.85f;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "Error: %s\nFile %s, line %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

__host__
void Graphpreproc(
	string const filename, 
	vector<int> &vertex_begin,
	int* edge)
{
	ifstream in_f;
	vector<int> t;
	vector<int>::iterator itr;

	int count = 0;
	int e = 0;
	in_f.open(filename.c_str(), ios::in);

	string line;
	char delim[3] = " \t";	//In most benchmarks, the delimiter is usually the space character or the tab character.
	char* pch;

	// Read the input graph line-by-line.
	while( !in_f.eof()) {
		getline(in_f, line);
		if( line[0] < '0' || line[0] > '9' )	// Skipping any line blank or starting with a character rather than a number.
			continue;
		char cstrLine[256];
		strcpy( cstrLine, line.c_str() );

		pch = strtok(cstrLine, delim);
		if( pch != NULL )
			t.push_back(atoi(pch));
		else
			continue;
		pch = strtok( NULL, delim );
		if( pch != NULL )
			edge[e++] = atoi(pch);
		else
			continue;
	}

	itr = t.begin();
	vertex_begin.push_back(0);
	for (int i = 0; i < vertex_num - 1; i++){
		while ((itr != t.end()) && (*itr == i)){
			count++;
			itr++;
		}
		count += vertex_begin.at(i);
		vertex_begin.push_back(count);
		count = 0;
	}
	vertex_begin.push_back(edge_num);
}

__host__
void greedy_ord(
	int* edge,
	int* trans)
{
    bool* vis = new bool[vertex_num];
    memset(vis, false, sizeof(bool)*vertex_num);
    
    int curr_pos = 0;
    
    for(int e = 0; e < edge_num; e++){
        if(!vis[edge[e]]){
            vis[edge[e]] = true;
            trans[edge[e]] = curr_pos;
            edge[e] = curr_pos++;
        }
        else
            edge[e] = trans[edge[e]];
    }
}

/*__global__ void kernel_vertex(
	int const vertex_num,
	const int* const vertex_begin,
	float* const values,
	float* const tmp)
{
	int n = blockDim.x * gridDim.x/32;    //total warp number
	int tid = threadIdx.x % 32;
	int wid = blockIdx.x * blockDim.x/32 + threadIdx.x/32;


	for(int i = wid; i < vertex_num; i += n){
		int degree = vertex_begin[i + 1] - vertex_begin[i];
		if(degree > 0){
			int loop_num = degree / 32;
			if(tid < degree){
				for(int j = 0; j <= loop_num; j++)
					atomicAdd(&tmp[i], values[vertex_begin[i] + 32*j + tid%degree]);
			}
			if(tid == 0){
				tmp[i] = PAGERANK_COEFFICIENT * tmp[i] + 1.0f - PAGERANK_COEFFICIENT;
				values[i] = tmp[i] / degree;
			}
		}
	}
}*/

__global__ void kernel_vertex(
	int const vertex_num,
	const int* const vertex_begin,
	const int* const edge,
	float* const values,
	float* const tmp)
{
	int n = blockDim.x * gridDim.x/32;    //total warp number
	int tid = threadIdx.x % 32;
	int wid = blockIdx.x * blockDim.x/32 + threadIdx.x/32;

	for(int i = wid; i < vertex_num; i += n){
		int degree = vertex_begin[i + 1] - vertex_begin[i];
		if(degree > 0){
			int loop_num = degree / 32;
			if(tid < degree){
				for(int j = 0; j <= loop_num; j++)
					atomicAdd(&tmp[i], values[edge[vertex_begin[i] + 32*j + tid%degree]]);
			}
			if(tid == 0){
				tmp[i] = PAGERANK_COEFFICIENT * tmp[i] + 1.0f - PAGERANK_COEFFICIENT;
				values[i] = tmp[i] / degree;
			}
		}
	}
}

int main(int argc, const char * argv[])
{
	if(argc < 4){
		cout << "parameter should be three!";
		return 0;
	}

	fn = argv[1];
	vertex_num = atoi(argv[2]);
	edge_num = atoi(argv[3]);

	vector<int> vertex_begin;
	vertex_begin.reserve(vertex_num + 1);

	int* edge = new int[edge_num];
	Graphpreproc(fn, vertex_begin, edge);

	int* trans = new int[vertex_num];
	greedy_ord(edge, trans);

	int * dev_vertex_begin;
	int * dev_edge;
	float * dev_values;
	float * dev_tmp;

	size_t memSize_R = (vertex_num + 1) * sizeof(int);
	size_t memSize_C = edge_num * sizeof(int);

	gpuErrchk(cudaMalloc(&dev_vertex_begin, memSize_R));
	gpuErrchk(cudaMemcpy(dev_vertex_begin, vertex_begin.data(), memSize_R, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&dev_edge, memSize_C));
	gpuErrchk(cudaMemcpy(dev_edge, edge, memSize_C, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&dev_values, memSize_C));
	gpuErrchk(cudaMemset(dev_values, 0.0, memSize_C));

	gpuErrchk(cudaMalloc(&dev_tmp, memSize_R));
	gpuErrchk(cudaMemset(dev_tmp, 0.0, memSize_R));

	int bn = 256;
	int tn = 128;

	cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	kernel_vertex<<<tn,bn>>>(
		vertex_num,
		dev_vertex_begin,
		dev_edge,
		dev_values,
		dev_tmp);


	cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time,start,stop);
    printf("time is %f\n",time);

    gpuErrchk(cudaFree(dev_values));
    gpuErrchk(cudaFree(dev_edge));
    gpuErrchk(cudaFree(dev_vertex_begin));
    gpuErrchk(cudaFree(dev_tmp));

    delete []edge;
    delete []trans;

    return 0;
}


