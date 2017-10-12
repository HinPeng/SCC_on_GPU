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

using namespace std;


const int N = 5;
const int M = 6;
const int BLOCK_SIZE = 256;

const unsigned MAX_SUB = 4294967295;

const int NUM_BANKS = 16;
const int LOG_NUM_BANKS = 4;
const string fn("test.txt");

unsigned **scanBlockSums;
unsigned numEltsAllocated = 0;
unsigned numLevelsAllocated = 0;

__device__  unsigned Mterminate;
//__managed__ unsigned numActiveThreads;
__device__ unsigned numActiveThreads;
__device__ unsigned *range;
__device__ unsigned *pivots;

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
unsigned *F, unsigned *X, unsigned *C, unsigned *Fu);
__global__
void BFSKernel2(size_t graphSize, unsigned *F, unsigned *X, unsigned *Fu);
__global__
void TRIMKernel(size_t graphSize,unsigned *V,unsigned *E,unsigned *eliminated);
__global__
void getActiveMaskTemp(size_t graphSize, unsigned *F, unsigned *activeMask);
__global__
void compactSIMD(size_t N, unsigned *prefixSums, unsigned *activeMask, size_t blockSize);

__host__
void setUInt(unsigned *address, unsigned value);
__host__
void Graphpreproc(const string filename, vector<unsigned> &VF, vector<unsigned> &EF, vector<unsigned> &VB, vector<unsigned> &EB);
__host__
void BFS(vector<unsigned> &V, vector<unsigned> &E, unsigned sourceVertex, vector<unsigned> &visited);
__host__
void TRIMMING(vector<unsigned> &V, vector<unsigned> &E, vector<unsigned> &eliminated);
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
	vector<unsigned> VF, EF, VB, EB, visF, visB,elimF,elimB;
	vector<unsigned>::iterator itr;
	VF.reserve(N + 1);
	EF.reserve(M);
	VB.reserve(N + 1);
	EB.reserve(M);
	Graphpreproc(fn, VF, EF, VB, EB);

	gpuErrchk(cudaMalloc(&range, N*sizeof(unsigned)));
	gpuErrchk(cudaMemset(range, 0, N*sizeof(unsigned)));

	unsigned st = 0;
	unsigned ed = 0;
	unsigned cur_max_sub = 0;
	
	while(true){
		for(unsigned i = st; i <= ed; i++){
			TRIMMING(VF,EF,elimF);
			TRIMMING(VB,EB,elimB);
			PIVOTS_SEL();
			BFS(VF, EF, pivot, visF);
			BFS(VB, EB, pivot, visB);
			unsigned j = 0;
			for(itr = r.begin();itr!=r.end();itr++){
				if(visF.at(j) == 1){
					if(visB.at(j) == 1)
						*itr = MAX_SUB;
					else
						*itr = ++cur_max_sub;
				}
				else{
					if(visB.at(j) == 1)
						*itr = ++cur_max_sub;
					else
						*itr
				}
			}
		}
	}
	/*BFS(VF, EF, 0, visF);
	BFS(VB, EB, 0, visB);
	for (itr = visF.begin(); itr != visF.end(); itr++)
		cout << *itr << ' ';
	for (itr = visB.begin(); itr != visB.end(); itr++)
		cout << *itr << ' ';*/
	for (itr = eliminated.begin(); itr != eliminated.end(); itr++)
		cout << *itr << ' ';
	cout << endl;
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
unsigned *F, unsigned *X, unsigned *Fu) {

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
void TRIMKernel(size_t graphSize,unsigned *V,unsigned *E,unsigned *eliminated)
{
	int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	bool elim;

	if((v < graphSize)&&(eliminated[v] == FALSE)){
		elim = true;
		if(V[v+1]>V[v])
			elim = false;
		if(elim == true){
			eliminated[v] = TRUE;
			Mterminate = FALSE;
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

	size_t blockStart = blockIdx.x * blockSize;
	// Vertex assigned to current thread
	size_t v = blockStart + threadIdx.x;

	if (v < N) {
		// Can possibly be accelerated by using shared memory
		if (prefixSums[v + 1] != prefixSums[v]) {
			activeMask[prefixSums[v]] = v;
		}
	}
}

__host__
void BFS(vector<unsigned> &V, vector<unsigned> &E, unsigned sourceVertex, vector<unsigned> &visited)
{
	assert(sizeof(unsigned) == 4);
	visited.clear();
	visited.resize(N);

	unsigned *d_V, *d_E;
	unsigned *d_F, *d_X, *d_Fu;
	unsigned *activeMask, *prefixSums;

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

	const size_t prefixSumGridSize =
		(N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	while (true) {

		// Terminate <- TRUE
		unsigned terminateHost = TRUE;

		gpuErrchk(cudaMemcpyToSymbol(Mterminate, &terminateHost, sizeof(unsigned)));

		// Kernel 1: need to assign ACTIVE vertices to SIMD lanes (threads)
		gpuErrchk(cudaMemcpyFromSymbol(&numActiveThreadsHost, numActiveThreads, sizeof(unsigned)));
		const size_t gridSizeK1 =
			(numActiveThreadsHost + BLOCK_SIZE - 1) / BLOCK_SIZE;

		// launch kernel 1
		BFSKernel1 << <gridSizeK1, BLOCK_SIZE >> > (N, activeMask, d_V, d_E, d_F, d_X, d_Fu);
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
			cudaMemcpy(&numActiveThreads, prefixSums + N, sizeof(unsigned), cudaMemcpyDeviceToDevice);

			const size_t gridSizeCompaction = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
			compactSIMD << <gridSizeCompaction, BLOCK_SIZE >> > (N, prefixSums, activeMask, BLOCK_SIZE);
			//gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());

			//gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}
	}

	// Download result

	//gpuErrchk(cudaMemcpy(distances.data(), d_C, memSize - sizeof(unsigned), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(visited.data(), d_X, memSize - sizeof(unsigned), cudaMemcpyDeviceToHost));

	// Free memory

	gpuErrchk(cudaFree(d_F));
	gpuErrchk(cudaFree(d_X));
	gpuErrchk(cudaFree(d_C));
	gpuErrchk(cudaFree(d_Fu));
	gpuErrchk(cudaFree(d_V));
	gpuErrchk(cudaFree(d_E));
	gpuErrchk(cudaFree(activeMask));
	deallocBlockSums();
	gpuErrchk(cudaFree(prefixSums));
}

__host__
void TRIMMING(vector<unsigned> &V, vector<unsigned> &E, vector<unsigned> &eliminated)
{
	eliminated.clear();
	eliminated.resize(N);

	unsigned *d_V, *d_E;
	unsigned *d_elim;

	size_t memSize = (N + 1) * sizeof(unsigned);
	size_t memSizeE = M * sizeof(unsigned);

	gpuErrchk(cudaMalloc(&d_V, memSize));
	gpuErrchk(cudaMemcpy(d_V, V.data(), memSize, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&d_E, memSizeE));
	gpuErrchk(cudaMemcpy(d_E, E.data(), memSizeE, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&d_elim, memSize));
	gpuErrchk(cudaMemset(d_elim, FALSE, memSize));

	while (true) {

		// Terminate <- TRUE
		unsigned terminateHost = TRUE;

		gpuErrchk(cudaMemcpyToSymbol(Mterminate, &terminateHost, sizeof(unsigned)));

		// Kernel 2: need to assign ALL vertices to SIMD lanes
		const size_t gridSizeK2 =
			(N + BLOCK_SIZE - 1) / BLOCK_SIZE;

		// launch kernel 2
		TRIMKernel << <gridSizeK2, BLOCK_SIZE >> > (N, d_V, d_E, d_elim);
		//gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());


		gpuErrchk(cudaMemcpyFromSymbol(&terminateHost, Mterminate, sizeof(unsigned)));

		if (terminateHost) {
			break;
		}
	}

	gpuErrchk(cudaMemcpy(eliminated.data(), d_elim, memSize - sizeof(unsigned), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_V));
	gpuErrchk(cudaFree(d_E));
	gpuErrchk(cudaFree(d_elim));
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