#include "kernel.h"

#define OFFSET(i, j, N) (i) * (N) + (j)
__global__ void naive_kernel(float *a, float *b, float *c, int N, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    for (int i = 0; i < K; i++)
        c[tid] += a[i] * b[OFFSET(i, tid, N)];
}

void naive(float *a, float *b, float *c, int N, int K) {
    const int BLOCK_SIZE = 32;
    naive_kernel<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, N, K);
}

__device__ int warpReduce(int val) {
    for (int i = 16; i >= 1; i /= 2)
        val += __shfl_xor_sync(0xffffffff, val, i);
    return val;
}

__global__ void warpReduce_kernel(float *a, float *b, float *c, int N, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;
    int warpSize = 32;

    int val = 0;
    for (int i = 0; i < K; i += warpSize) 
        val += a[laneId + i] * b[OFFSET(laneId + i, warpId, N)];
    c[warpId] = warpReduce(val);
}

void warpReduce(float *a, float *b, float *c, int N, int K) {
    const int NUM_THREAD = 1024;
    const int WARP_SIZE = 32;
    
    warpReduce_kernel<<<N / (NUM_THREAD / WARP_SIZE), NUM_THREAD>>>(a, b, c, N, K);
}

int main() {
    int N = 64, K = 64;

    size_t size_a = sizeof(float) * K;
    size_t size_b = sizeof(float) * K * N;
    size_t size_c = sizeof(float) * N;

    float *a = (float *)malloc(size_a);
    float *b = (float *)malloc(size_b);
    float *c = (float *)malloc(size_c);

    for (int i = 0; i < K; i++) a[i] = 1;
    for (int i = 0; i < K * N; i++) b[i] = 1;

    float *da, *db, *dc;
    cudaMalloc(&da, size_a);
    cudaMalloc(&db, size_b);
    cudaMalloc(&dc, size_c);

    cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice);

    // naive(da, db, dc, N, K);
    warpReduce(da, db, dc, N, K);

    cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        std::cout << c[i] << " ";

    return 0;
}