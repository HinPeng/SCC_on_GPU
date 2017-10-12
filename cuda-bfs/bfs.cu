#include "bfs.hpp"
#include "bfs_kernels.cuh"
#include "compaction.cuh"
#include <stdio.h>

extern __device__  unsigned terminate_d;
extern __device__ unsigned numActiveThreads;

__host__
void setUInt(unsigned *address, unsigned value) {
    gpuErrchk(cudaMemcpy(address, &value, sizeof(unsigned), cudaMemcpyHostToDevice));
}

// If you are going to debug
__global__
void output(int N, unsigned *ptr) {
    for (int i = 0; i < N; ++i) {
        printf("%u ", ptr[i]);
    }
    printf("\n");
}

__host__
void BFS(vector<unsigned> &V, vector<unsigned> &E, unsigned sourceVertex, std::vector<unsigned> & distances) {

    assert(sizeof(unsigned) == 4);
    
    distances.clear();
    distances.resize(N);

    // Memory allocation and setup

    unsigned *d_V, *d_E;
    unsigned *d_F, *d_X, *d_C, *d_Fu;
    unsigned *activeMask, *prefixSums;

    size_t memSize = (N + 1) * sizeof(unsigned);
    
    gpuErrchk(cudaMalloc(&d_F, memSize));
    gpuErrchk(cudaMemset(d_F, FALSE, memSize));
    setUInt(d_F + sourceVertex, TRUE); // add source to frontier

    gpuErrchk(cudaMalloc(&d_X, memSize));
    gpuErrchk(cudaMemset(d_X, FALSE, memSize));
    setUInt(d_X + sourceVertex, TRUE); // set source as visited

    gpuErrchk(cudaMalloc(&d_C, memSize));
    gpuErrchk(cudaMemset(d_C, 255, memSize)); // set "infinite" distance
    setUInt(d_C + sourceVertex, FALSE); // set zero distance to source

    gpuErrchk(cudaMalloc(&d_Fu, memSize));
    gpuErrchk(cudaMemset(d_Fu, FALSE, memSize));

    gpuErrchk(cudaMalloc(&d_V, memSize));
    gpuErrchk(cudaMemcpy(d_V, V.data(), memSize, cudaMemcpyHostToDevice));

    size_t memSizeE = M * sizeof(unsigned);
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
        gpuErrchk(cudaMemcpyToSymbol(terminate_d, &terminateHost, sizeof(unsigned)));

        // Kernel 1: need to assign ACTIVE vertices to SIMD lanes (threads)
        //gpuErrchk(cudaMemcpyFromSymbol(&numActiveThreadsHost, numActiveThreads, sizeof(unsigned)));
        const size_t gridSizeK1 = 
            (numActiveThreadsHost + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // launch kernel 1
        BFSKernel1 <<<gridSizeK1, BLOCK_SIZE>>> (N, activeMask, d_V, d_E, d_F, d_X, d_C, d_Fu);
        //gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // Kernel 2: need to assign ALL vertices to SIMD lanes
        const size_t gridSizeK2 =
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // launch kernel 2
        BFSKernel2 <<<gridSizeK2, BLOCK_SIZE>>> (N, d_F, d_X, d_Fu);
        //gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaMemcpyFromSymbol(&terminateHost, terminate_d, sizeof(unsigned)));

        if (terminateHost) {
            break;
        } else {
            // Get prefix sums of F
            prescanArray(prefixSums, d_F, N + 1);
            //cudaMemcpy(&numActiveThreads, prefixSums + N, sizeof(unsigned), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&numActiveThreadsHost, prefixSums + N, sizeof(unsigned), cudaMemcpyDeviceToHost);
            gpuErrchk(cudaMemcpyToSymbol(numActiveThreads, &numActiveThreadsHost, sizeof(unsigned)));


            /*gpuErrchk(cudaMemcpyFromSymbol(&numActiveThreadsHost, numActiveThreads, sizeof(unsigned)));
            printf("%u\n", numActiveThreadsHost);*/
            
            const size_t gridSizeCompaction = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            compactSIMD <<<gridSizeCompaction, BLOCK_SIZE>>> (N, prefixSums, activeMask, BLOCK_SIZE);
            //gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            //gpuErrchk(cudaPeekAtLastError());
            //gpuErrchk(cudaDeviceSynchronize());
        }
    }

    // Download result

    gpuErrchk(cudaMemcpy(distances.data(), d_X, memSize-sizeof(unsigned), cudaMemcpyDeviceToHost));

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

