#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iterator>
#include <algorithm>
#include <assert.h>
#include <time.h>

using namespace std;


const int N = 2394385;
const int M = 5021410;
const int BLOCK_SIZE = 256;

const unsigned MAX_NUM = 4294967295;

const int NUM_BANKS = 16;
const int LOG_NUM_BANKS = 4;
const string fn("WikiTalk.txt");
//const string fn("soc-LiveJournal1.txt");

unsigned **scanBlockSums;
unsigned numEltsAllocated = 0;
unsigned numLevelsAllocated = 0;

__device__  unsigned Mterminate;
//__managed__ unsigned numActiveThreads;
__device__ unsigned numActiveThreads;
__device__ unsigned *range;
__device__ unsigned pivot;

#define FALSE 0u
#define  TRUE 1u

#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "Error: %s\nFile %s, line %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

inline
bool isPowerOfTwo(int n) {
	return (n & (n - 1)) == 0;
}

inline
int floorPow2(int n) {
	int exp;
	frexp((float)n, &exp);
	return 1 << (exp - 1);
}

template <bool isNP2>
__device__
void loadSharedChunkFromMem(unsigned *s_data, const unsigned *idata, int n, int baseIndex, int& ai, int& bi, int& mem_ai, int& mem_bi, int& bankOffsetA, int& bankOffsetB);
template <bool isNP2>
__device__
void storeSharedChunkToMem(unsigned* odata, const unsigned* s_data, int n, int ai, int bi, int mem_ai, int mem_bi, int bankOffsetA, int bankOffsetB);
template <bool storeSum>
__device__
void clearLastElement(unsigned* s_data, unsigned *blockSums, int blockIndex);
__device__
unsigned int buildSum(unsigned *s_data);
__device__
void scanRootToLeaves(unsigned *s_data, unsigned int stride);
template <bool storeSum>
__device__
void prescanBlock(unsigned *data, int blockIndex, unsigned *blockSums);
template <bool storeSum, bool isNP2>
__global__
void prescan(unsigned *odata, const unsigned *idata, unsigned *blockSums, int n, int blockIndex, int baseIndex);
__global__
void uniformAdd(unsigned *data, unsigned *uniforms, int n, int blockOffset, int baseIndex);

__global__
void BFSKernel1(
size_t graphSize, unsigned *activeMask, unsigned *V, unsigned *E,
unsigned *F, unsigned *X,unsigned *Fu);
__global__
void BFSKernel2(size_t graphSize, unsigned *F, unsigned *X, unsigned *Fu);
__global__
void TRIMKernel(size_t graphSize,unsigned subnum, unsigned *V,unsigned *E);
__global__
void getActiveMaskTemp(size_t graphSize, unsigned *F, unsigned *activeMask);
__global__
void compactSIMD(size_t N, unsigned *prefixSums, unsigned *activeMask, size_t blockSize);
__global__
void PIVOTS_SEL_Kernel(size_t graphSize,unsigned subnum);
__global__
void UpdateKernel(unsigned cur_max_sub,unsigned subnum,unsigned *visF,unsigned *visB,unsigned *scc);

__host__
void setUInt(unsigned *address, unsigned value);
__host__
void Graphpreproc(const string filename, vector<unsigned> &VF, vector<unsigned> &EF, vector<unsigned> &VB, vector<unsigned> &EB);
__host__
void BFS(vector<unsigned> &V, vector<unsigned> &E, unsigned sourceVertex,vector<unsigned> &visited);
__host__
void TRIMMING(unsigned index_s,unsigned index_e, vector<unsigned> &V, vector<unsigned> &E);
__host__
void PIVOTS_SEL(unsigned subnum,unsigned &pivot_h,unsigned &termin);
__host__
void Update(unsigned cur_max_sub,unsigned subnum,vector<unsigned> &visF,vector<unsigned> &visB,vector<unsigned> &scc,unsigned &termin);
__host__
void preallocBlockSums(unsigned maxNumElements);
__host__
void deallocBlockSums();
__host__
void prescanArrayRecursive(unsigned *outArray, const unsigned *inArray, int numElements, int level);
__host__
void prescanArray(unsigned *outArray, unsigned *inArray, int numElements);

int main()
{
	vector<unsigned> VF, EF, VB, EB, visF, visB, scc;
	vector<unsigned>::iterator itr;

	ofstream out("out1.txt");
	if(!out){  
    	cout << "Unable to open outfile";  
        exit(1); // terminate with error  
    }
  

	bool flag = true;

	unsigned pivot_h;
	unsigned termin;

	unsigned *d_r;

	VF.reserve(N + 1);
	EF.reserve(M);
	VB.reserve(N + 1);
	EB.reserve(M);
	Graphpreproc(fn, VF, EF, VB, EB);

	unsigned st = 0;
	unsigned ed = 0;
	unsigned cur_max_sub = 0;

	long trim = 0;
	long bfs = 0;
	long pivsel = 0;
	long upd = 0;

	clock_t start=clock();

	gpuErrchk(cudaMalloc(&d_r, N*sizeof(unsigned)));
	gpuErrchk(cudaMemset(d_r, FALSE, N*sizeof(unsigned)));
	gpuErrchk(cudaMemcpyToSymbol(range, &d_r, sizeof(unsigned *),size_t(0), cudaMemcpyHostToDevice));

	while(flag){
		flag = false;
		clock_t start_trim=clock();
		TRIMMING(st,ed,VF,EF);
		TRIMMING(st,ed,VB,EB);
		clock_t finish_trim=clock();
		trim += (finish_trim - start_trim);
		for(unsigned i = st; i <= ed; i++){
			clock_t start_pivsel=clock();
			PIVOTS_SEL(i,pivot_h,termin);
			clock_t finish_pivsel=clock();
			pivsel += (finish_pivsel - start_pivsel);
			if(termin)
				continue;

			clock_t start_bfs=clock();
			BFS(VF, EF, pivot_h, visF);
			BFS(VB, EB, pivot_h, visB);
			clock_t finish_bfs=clock();
			bfs += (finish_bfs - start_bfs);

			clock_t start_upd=clock();
			Update(cur_max_sub,i,visF,visB,scc,termin);
			clock_t finish_upd=clock();
			upd += (finish_upd - start_upd);

			unsigned j = 0;
			for(itr = scc.begin();itr!=scc.end();itr++){
				if(*itr == 1){
					j++;
				}
			}
			out<<j<<endl;
			cur_max_sub += 3;
			if(termin == FALSE)
				flag = true;
		}
		st = ed + 1;
		ed = cur_max_sub;
	}
	clock_t finish=clock();
	printf("time elapsed:%.2fs\n",(double)(finish-start)/1.0e6);
	printf("TRIM time elapsed:%.2fs\n",(double)(trim)/1.0e6);
	printf("PIVSEL time elapsed:%.2fs\n",(double)(pivsel)/1.0e6);
	printf("BFS time elapsed:%.2fs\n",(double)(bfs)/1.0e6);
	printf("UPD time elapsed:%.2fs\n",(double)(upd)/1.0e6);
	printf("%u\n",ed -3);

	out.close();
	return 0;
}

__host__
void setUInt(unsigned *address, unsigned value) {
	gpuErrchk(cudaMemcpy(address, &value, sizeof(unsigned), cudaMemcpyHostToDevice));
}

__host__
void Graphpreproc(const string filename, vector<unsigned> &VF, vector<unsigned> &EF, vector<unsigned> &VB, vector<unsigned> &EB)
{
	ifstream in_f;
	vector<unsigned> t;
	vector<unsigned>::iterator itr;
	multimap<const unsigned, unsigned> m;
	multimap<const unsigned, unsigned>::iterator mitr;
	unsigned count = 0;
	in_f.open(filename.c_str(), ios::in);
	while (!in_f.eof()){
		string temp, s1, s2;
		stringstream ss1, ss2;
		unsigned t1, t2;
		getline(in_f, temp);
		if(temp.length() == 0)
			continue;
		if (*(temp.begin()) == '#')
			continue;
		s1 = string(temp, 0, temp.find_first_of('\t'));
		s2 = string(temp, temp.find_first_not_of('\t', temp.find_first_of('\t')), temp.find_last_not_of('\t'));
		ss1 << s1;
		ss1 >> t1;
		ss2 << s2;
		ss2 >> t2;
		t.push_back(t1);
		m.insert(make_pair(t2, t1));
		EF.push_back(t2);
	}

	itr = t.begin();
	VF.push_back(0);
	for (int i = 0; i < N - 1; i++){
		while ((itr != t.end()) && (*itr == i)){
			count++;
			itr++;
		}
		count += VF.at(i);
		VF.push_back(count);
		count = 0;
	}
	VF.push_back(M);

	mitr = m.begin();
	VB.push_back(0);
	for (int i = 0; i < N - 1; i++){
		while ((mitr != m.end()) && ((mitr->first) == i)){
			count++;
			mitr++;
		}
		count += VB.at(i);
		VB.push_back(count);
		count = 0;
	}
	VB.push_back(M);
	for (mitr = m.begin(); mitr != m.end(); mitr++){
		EB.push_back(mitr->second);
	}
}

__global__
void BFSKernel1(
size_t graphSize, unsigned *activeMask, unsigned *V, unsigned *E,
unsigned *F, unsigned *X,unsigned *Fu) {

	unsigned activeMaskIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	// If vertex is active at current iteration
	if (activeMaskIdx < numActiveThreads) {

		unsigned v = activeMask[activeMaskIdx];

		// Remove v from current frontier
		F[v] = FALSE;

		// Iterate over v's neighbors
		for (unsigned edge = V[v]; edge < V[v + 1]; ++edge) {
			unsigned neighbor = E[edge];

			// If neighbor wasn't visited
			if(range[v] == range[neighbor]){
				if (X[neighbor] == FALSE){
					//C[neighbor] = C[v] + 1;
					Fu[neighbor] = TRUE;
				}
			}
		}
	}
}

__global__
void BFSKernel2(size_t graphSize, unsigned *F, unsigned *X, unsigned *Fu) {

	int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	// If vertex v exists and has recently joined the frontier
	if (v < graphSize && Fu[v]) {
		// Copy the new frontier into F
		F[v] = TRUE;
		// Set v as visited
		X[v] = TRUE;
		// Clean up the new frontier
		Fu[v] = FALSE;

		Mterminate = FALSE;
	}
}

__global__
void TRIMKernel(size_t graphSize,unsigned subnum, unsigned *V,unsigned *E)
{
	int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	bool elim;

	if((v < graphSize)&&(range[v] == subnum)){
		elim = true;
		for (unsigned edge = V[v]; edge < V[v + 1]; ++edge) {
			unsigned neighbor = E[edge];
			if(range[neighbor] == subnum)
				elim = false;
		}
		if(elim == true){
			range[v] = MAX_NUM;
			Mterminate = FALSE;
		}
	}
}

__global__
void PIVOTS_SEL_Kernel(size_t graphSize,unsigned subnum)
{
	int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	if((v < graphSize)&&(range[v] == subnum)){
		pivot = v;
		Mterminate = FALSE;
	}
}

__global__
void UpdateKernel(unsigned cur_max_sub,unsigned subnum,unsigned *visF,unsigned *visB,unsigned *scc)
{
	int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	if(range[v] == subnum){
		if(visF[v] == TRUE){
			if(visB[v] == TRUE){
				scc[v] = TRUE;
				range[v] = MAX_NUM;
			}
			else if(visB[v] == FALSE){
				range[v] = cur_max_sub + 1;
				Mterminate = FALSE;
			}
		}
		else if(visF[v] == FALSE){
			if(visB[v] == TRUE){
				range[v] = cur_max_sub + 2;
				Mterminate = FALSE;
			}
			else if(visB[v] == FALSE){
				range[v] = cur_max_sub + 3;
				Mterminate = FALSE;
			}
		}
	}
}

// Very slow but correct "active mask" calculation; for debugging
__global__
void getActiveMaskTemp(size_t graphSize, unsigned *F, unsigned *activeMask) {

	numActiveThreads = 0;
	for (int i = 0; i < graphSize; ++i) {
		if (F[i]) {
			activeMask[numActiveThreads] = i;
			++numActiveThreads;
		}
	}
}

__global__
void compactSIMD(size_t N, unsigned *prefixSums, unsigned *activeMask, size_t blockSize) {

	size_t v = blockIdx.x * blockSize + threadIdx.x;

	if (v < N) {
		// Can possibly be accelerated by using shared memory
		if (prefixSums[v + 1] != prefixSums[v]) {
			activeMask[prefixSums[v]] = v;
		}
	}
}

__host__
void BFS(vector<unsigned> &V, vector<unsigned> &E, unsigned sourceVertex,vector<unsigned> &visited)
{
	assert(sizeof(unsigned) == 4);
	visited.clear();
	visited.resize(N);

	unsigned *d_V, *d_E;
	unsigned *d_F, *d_X, *d_Fu;
	unsigned *activeMask, *prefixSums;
	//unsigned **prefixSums;
	size_t memSize = (N + 1) * sizeof(unsigned);
	size_t memSizeE = M * sizeof(unsigned);

	gpuErrchk(cudaMalloc(&d_F, memSize));
	gpuErrchk(cudaMemset(d_F, FALSE, memSize));
	setUInt(d_F + sourceVertex, TRUE); // add source to frontier

	gpuErrchk(cudaMalloc(&d_X, memSize));
	gpuErrchk(cudaMemset(d_X, FALSE, memSize));
	setUInt(d_X + sourceVertex, TRUE); // set source as visited

	//gpuErrchk(cudaMalloc(&d_C, memSize));
	//gpuErrchk(cudaMemset(d_C, 255, memSize)); // set "infinite" distance
	//setUInt(d_C + sourceVertex, FALSE); // set zero distance to source

	gpuErrchk(cudaMalloc(&d_Fu, memSize));
	gpuErrchk(cudaMemset(d_Fu, FALSE, memSize));

	gpuErrchk(cudaMalloc(&d_V, memSize));
	gpuErrchk(cudaMemcpy(d_V, V.data(), memSize, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&d_E, memSizeE));
	gpuErrchk(cudaMemcpy(d_E, E.data(), memSizeE, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&activeMask, memSize));
	setUInt(activeMask + 0, sourceVertex); // set thread #source as active

	unsigned numActiveThreadsHost = 1;
	gpuErrchk(cudaMemcpyToSymbol(numActiveThreads, &numActiveThreadsHost, sizeof(unsigned)));

	gpuErrchk(cudaMalloc(&prefixSums, memSize));
	preallocBlockSums(N + 1);

	// Main loop

	/*const size_t prefixSumGridSize =
		(N + BLOCK_SIZE - 1) / BLOCK_SIZE;*/

	while (true) {

		// Terminate <- TRUE
		unsigned terminateHost = TRUE;

		gpuErrchk(cudaMemcpyToSymbol(Mterminate, &terminateHost, sizeof(unsigned)));

		// Kernel 1: need to assign ACTIVE vertices to SIMD lanes (threads)
		gpuErrchk(cudaMemcpyFromSymbol(&numActiveThreadsHost, numActiveThreads, sizeof(unsigned)));
		const size_t gridSizeK1 =
			(numActiveThreadsHost + BLOCK_SIZE - 1) / BLOCK_SIZE;

		// launch kernel 1
		BFSKernel1 <<<gridSizeK1, BLOCK_SIZE >>> (N, activeMask, d_V, d_E, d_F, d_X,d_Fu);
		//gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		// Kernel 2: need to assign ALL vertices to SIMD lanes
		const size_t gridSizeK2 =
			(N + BLOCK_SIZE - 1) / BLOCK_SIZE;

		// launch kernel 2
		BFSKernel2 << <gridSizeK2, BLOCK_SIZE >> > (N, d_F, d_X, d_Fu);
		//gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());


		gpuErrchk(cudaMemcpyFromSymbol(&terminateHost, Mterminate, sizeof(unsigned)));

		if (terminateHost) {
			break;
		}
		else {
			// Get prefix sums of F
			prescanArray(prefixSums, d_F, N + 1);
			//cudaMemcpy(&numActiveThreads, prefixSums + N, sizeof(unsigned), cudaMemcpyDeviceToDevice);
			cudaMemcpy(&numActiveThreadsHost, prefixSums + N, sizeof(unsigned), cudaMemcpyDeviceToHost);
            gpuErrchk(cudaMemcpyToSymbol(numActiveThreads, &numActiveThreadsHost, sizeof(unsigned)));

			const size_t gridSizeCompaction = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
			compactSIMD <<<gridSizeCompaction, BLOCK_SIZE >>> (N, prefixSums, activeMask, BLOCK_SIZE);
			//gpuErrchk(cudaPeekAtLastError());
			//getActiveMaskTemp<<<1,1>>>(N,d_F,activeMask);
			gpuErrchk(cudaDeviceSynchronize());

			//gpuErrchk(cudaPeekAtLastError());
		}
	}

	// Download result

	//gpuErrchk(cudaMemcpy(distances.data(), d_C, memSize - sizeof(unsigned), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(visited.data(), d_X, memSize - sizeof(unsigned), cudaMemcpyDeviceToHost));

	// Free memory

	gpuErrchk(cudaFree(d_F));
	gpuErrchk(cudaFree(d_X));
	gpuErrchk(cudaFree(d_Fu));
	gpuErrchk(cudaFree(d_V));
	gpuErrchk(cudaFree(d_E));
	gpuErrchk(cudaFree(activeMask));
	deallocBlockSums();
	gpuErrchk(cudaFree(prefixSums));
}

__host__
void TRIMMING(unsigned index_s,unsigned index_e,vector<unsigned> &V, vector<unsigned> &E)
{
	unsigned *d_V, *d_E;

	size_t memSize = (N + 1) * sizeof(unsigned);
	size_t memSizeE = M * sizeof(unsigned);

	gpuErrchk(cudaMalloc(&d_V, memSize));
	gpuErrchk(cudaMemcpy(d_V, V.data(), memSize, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&d_E, memSizeE));
	gpuErrchk(cudaMemcpy(d_E, E.data(), memSizeE, cudaMemcpyHostToDevice));

	const size_t gridSizeK2 =
			(N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	cudaStream_t stream[index_e-index_s+1];
	for (int i = 0; i < index_e-index_s+1; ++i)
    	cudaStreamCreate(&stream[i]);

    for (int i = 0; i < index_e-index_s+1; ++i){
		while (true) {

		// Terminate <- TRUE
			unsigned terminateHost = TRUE;

			gpuErrchk(cudaMemcpyToSymbol(Mterminate, &terminateHost, sizeof(unsigned)));

		// Kernel 2: need to assign ALL vertices to SIMD lanes
		
		// launch kernel 2
			TRIMKernel <<<gridSizeK2, BLOCK_SIZE, 0, stream[i] >>> (N, index_s+i,d_V, d_E);
		//gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());


			gpuErrchk(cudaMemcpyFromSymbol(&terminateHost, Mterminate, sizeof(unsigned)));

			if (terminateHost) {
			break;
			}
		}
	}

	for (int i = 0; i < index_e-index_s+1; ++i)
    	cudaStreamDestroy(stream[i]);

	gpuErrchk(cudaFree(d_V));
	gpuErrchk(cudaFree(d_E));
}

__host__
void PIVOTS_SEL(unsigned subnum,unsigned &pivot_h,unsigned &termin)
{
	const size_t gridSizeK2 =
			(N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	termin = TRUE;

	gpuErrchk(cudaMemcpyToSymbol(Mterminate, &termin, sizeof(unsigned)));

	PIVOTS_SEL_Kernel<<<gridSizeK2, BLOCK_SIZE>>>(N,subnum);

	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpyFromSymbol(&pivot_h, pivot, sizeof(unsigned)));
	gpuErrchk(cudaMemcpyFromSymbol(&termin, Mterminate, sizeof(unsigned)));
}

__host__
void Update(unsigned cur_max_sub,unsigned subnum,vector<unsigned> &visF,vector<unsigned> &visB,vector<unsigned> &scc,unsigned &termin)
{
	scc.clear();
	scc.resize(N);

	unsigned *d_vf,*d_vb;
	unsigned *d_scc;

	size_t memSize = N * sizeof(unsigned);

	gpuErrchk(cudaMalloc(&d_vf, memSize));
	gpuErrchk(cudaMemcpy(d_vf, visF.data(), memSize, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&d_vb, memSize));
	gpuErrchk(cudaMemcpy(d_vb, visB.data(), memSize, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&d_scc, memSize));
	gpuErrchk(cudaMemset(d_scc, FALSE, memSize));

	// Terminate <- TRUE
	termin = TRUE;

	gpuErrchk(cudaMemcpyToSymbol(Mterminate, &termin, sizeof(unsigned)));

	const size_t gridSizeK2 =
			(N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	UpdateKernel<<<gridSizeK2, BLOCK_SIZE>>>(cur_max_sub,subnum,d_vf,d_vb,d_scc);

	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(scc.data(), d_scc, memSize, cudaMemcpyDeviceToHost));

	gpuErrchk(cudaMemcpyFromSymbol(&termin, Mterminate, sizeof(unsigned)));

	gpuErrchk(cudaFree(d_vf));
	gpuErrchk(cudaFree(d_vb));
	gpuErrchk(cudaFree(d_scc));
}

__host__
void preallocBlockSums(unsigned maxNumElements) {
	numEltsAllocated = maxNumElements;

	unsigned blockSize = BLOCK_SIZE;
	unsigned numElts = maxNumElements;

	int level = 0;

	do {
		unsigned numBlocks =
			max(1, (int)ceil((float)numElts / (2.f * blockSize)));
		if (numBlocks > 1) {
			level++;
		}
		numElts = numBlocks;
	} while (numElts > 1);

	scanBlockSums = (unsigned**)malloc(level * sizeof(unsigned*));
	numLevelsAllocated = level;

	numElts = maxNumElements;
	level = 0;

	do {
		unsigned numBlocks =
			max(1, (int)ceil((float)numElts / (2.f * blockSize)));
		if (numBlocks > 1) {
			gpuErrchk(cudaMalloc(&scanBlockSums[level++], numBlocks * sizeof(unsigned)));
		}
		numElts = numBlocks;
	} while (numElts > 1);
}

__host__
void deallocBlockSums() {
	for (unsigned i = 0; i < numLevelsAllocated; i++) {
		cudaFree(scanBlockSums[i]);
	}

	free(scanBlockSums);

	scanBlockSums = 0;
	numEltsAllocated = 0;
	numLevelsAllocated = 0;
}

__host__
void prescanArrayRecursive(unsigned *outArray,
const unsigned *inArray,
int numElements,
int level) {

	unsigned blockSize = BLOCK_SIZE;
	unsigned numBlocks =
		max(1, (int)ceil((float)numElements / (2.f * blockSize)));
	unsigned numThreads;

	if (numBlocks > 1)
		numThreads = blockSize;
	else if (isPowerOfTwo(numElements))
		numThreads = numElements / 2;
	else
		numThreads = floorPow2(numElements);

	unsigned numEltsPerBlock = numThreads * 2;

	unsigned numEltsLastBlock =
		numElements - (numBlocks - 1) * numEltsPerBlock;
	unsigned numThreadsLastBlock = max(1u, numEltsLastBlock / 2);
	unsigned np2LastBlock = 0;
	unsigned sharedMemLastBlock = 0;

	if (numEltsLastBlock != numEltsPerBlock) {
		np2LastBlock = 1;

		if (!isPowerOfTwo(numEltsLastBlock))
			numThreadsLastBlock = floorPow2(numEltsLastBlock);

		unsigned extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
		sharedMemLastBlock =
			sizeof(unsigned)* (2 * numThreadsLastBlock + extraSpace);
	}

	// Avoid shared memory bank conflicts
	unsigned extraSpace = numEltsPerBlock / NUM_BANKS;
	unsigned sharedMemSize =
		sizeof(unsigned)* (numEltsPerBlock + extraSpace);

	dim3 grid(max(1u, numBlocks - np2LastBlock), 1, 1);
	dim3 threads(numThreads, 1, 1);

	// Main action

	if (numBlocks > 1) {
		prescan<true, false> << < grid, threads, sharedMemSize >> > (
			outArray, inArray, scanBlockSums[level], numThreads * 2, 0, 0);

		if (np2LastBlock) {
			prescan<true, true> << < 1, numThreadsLastBlock, sharedMemLastBlock >> > (
				outArray, inArray, scanBlockSums[level], numEltsLastBlock,
				numBlocks - 1, numElements - numEltsLastBlock);
		}

		prescanArrayRecursive(scanBlockSums[level], scanBlockSums[level], numBlocks, level + 1);

		uniformAdd << < grid, threads >> > (
			outArray, scanBlockSums[level], numElements - numEltsLastBlock, 0, 0);

		if (np2LastBlock) {
			uniformAdd << <1, numThreadsLastBlock >> > (
				outArray, scanBlockSums[level], numEltsLastBlock,
				numBlocks - 1, numElements - numEltsLastBlock);
		}
	}
	else if (isPowerOfTwo(numElements)) {
		prescan<false, false> << <grid, threads, sharedMemSize >> > (
			outArray, inArray, 0, numThreads * 2, 0, 0);
	}
	else {
		prescan<false, true> << <grid, threads, sharedMemSize >> > (
			outArray, inArray, 0, numElements, 0, 0);
	}
}

__host__
void prescanArray(unsigned *outArray, unsigned *inArray, int numElements) {
	prescanArrayRecursive(outArray, inArray, numElements, 0);
}

template <bool isNP2>
__device__ void loadSharedChunkFromMem(unsigned *s_data,
	const unsigned *idata,
	int n, int baseIndex,
	int& ai, int& bi,
	int& mem_ai, int& mem_bi,
	int& bankOffsetA, int& bankOffsetB) {
	int thid = threadIdx.x;
	mem_ai = baseIndex + threadIdx.x;
	mem_bi = mem_ai + blockDim.x;

	ai = thid;
	bi = thid + blockDim.x;

	bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	s_data[ai + bankOffsetA] = idata[mem_ai];

	if (isNP2) {
		s_data[bi + bankOffsetB] = (bi < n) ? idata[mem_bi] : 0;
	}
	else {
		s_data[bi + bankOffsetB] = idata[mem_bi];
	}
}

template <bool isNP2>
__device__
void storeSharedChunkToMem(unsigned* odata,
const unsigned* s_data,
int n,
int ai, int bi,
int mem_ai, int mem_bi,
int bankOffsetA, int bankOffsetB) {
	__syncthreads();

	odata[mem_ai] = s_data[ai + bankOffsetA];
	if (isNP2) {
		if (bi < n)
			odata[mem_bi] = s_data[bi + bankOffsetB];
	}
	else {
		odata[mem_bi] = s_data[bi + bankOffsetB];
	}
}

template <bool storeSum>
__device__
void clearLastElement(unsigned* s_data,
unsigned *blockSums,
int blockIndex) {
	if (threadIdx.x == 0) {
		int index = (blockDim.x << 1) - 1;
		index += CONFLICT_FREE_OFFSET(index);

		if (storeSum) {
			blockSums[blockIndex] = s_data[index];
		}

		s_data[index] = 0;
	}
}

__device__
unsigned int buildSum(unsigned *s_data) {
	unsigned int thid = threadIdx.x;
	unsigned int stride = 1;

	for (int d = blockDim.x; d > 0; d >>= 1) {
		__syncthreads();

		if (thid < d) {
			int i = __mul24(__mul24(2, stride), thid);
			int ai = i + stride - 1;
			int bi = ai + stride;

			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			s_data[bi] += s_data[ai];
		}

		stride *= 2;
	}

	return stride;
}

__device__
void scanRootToLeaves(unsigned *s_data, unsigned int stride) {
	unsigned int thid = threadIdx.x;

	for (int d = 1; d <= blockDim.x; d *= 2) {
		stride >>= 1;

		__syncthreads();

		if (thid < d)
		{
			int i = __mul24(__mul24(2, stride), thid);
			int ai = i + stride - 1;
			int bi = ai + stride;

			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			unsigned t = s_data[ai];
			s_data[ai] = s_data[bi];
			s_data[bi] += t;
		}
	}
}

template <bool storeSum>
__device__
void prescanBlock(unsigned *data, int blockIndex, unsigned *blockSums) {
	int stride = buildSum(data);
	clearLastElement<storeSum>(data, blockSums,
		(blockIndex == 0) ? blockIdx.x : blockIndex);
	scanRootToLeaves(data, stride);
}

template <bool storeSum, bool isNP2>
__global__
void prescan(unsigned *odata,
const unsigned *idata,
unsigned *blockSums,
int n,
int blockIndex,
int baseIndex) {
	int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
	extern __shared__ unsigned s_data[];

	loadSharedChunkFromMem<isNP2>(s_data, idata, n,
		(baseIndex == 0) ?
		__mul24(blockIdx.x, (blockDim.x << 1)) : baseIndex,
		ai, bi, mem_ai, mem_bi,
		bankOffsetA, bankOffsetB);

	prescanBlock<storeSum>(s_data, blockIndex, blockSums);

	storeSharedChunkToMem<isNP2>(odata, s_data, n,
		ai, bi, mem_ai, mem_bi,
		bankOffsetA, bankOffsetB);
}

__global__
void uniformAdd(unsigned *data,
unsigned *uniforms,
int n,
int blockOffset,
int baseIndex) {
	__shared__ unsigned uni;
	if (threadIdx.x == 0)
		uni = uniforms[blockIdx.x + blockOffset];

	unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x;

	__syncthreads();

	data[address] += uni;
	data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}