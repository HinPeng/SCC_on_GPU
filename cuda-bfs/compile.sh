set -e
nvcc  -arch=sm_35 -dc bfs_kernels.cu bfs.cu compaction.cu
nvcc  -arch=sm_35 -dlink bfs_kernels.o bfs.o compaction.o -o dlink.o
g++ bfs_kernels.o bfs.o compaction.o dlink.o main.cpp -lcudart -L/usr/local/cuda/lib64  -o main
