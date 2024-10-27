#include "kernel.h"

// we should execute blockSize threads
template<int blockSize>
__global__ void reduce_nonDivergent(float *src, float *dst) {
    int tid = threadIdx.x;
    int offset = blockIdx.x * blockSize + threadIdx.x;

    // copy
    __shared__ float shared[blockSize];
    shared[tid] = src[offset];
    __syncthreads();

    for (int i = 1; i < blockSize; i <<= 1) { 
        int idx = tid * i * 2;
        if (idx < blockSize) {
            shared[idx] += shared[idx + i];
        }
        __syncthreads();
    }

    // write back
    if (tid == 0) {
        dst[blockIdx.x] = shared[0];
    }
}

template<int blockSize>
__global__ void reduce_bankFree(float *src, float *dst) {
    int tid = threadIdx.x;
    int offset = blockIdx.x * blockSize + threadIdx.x;

    // copy
    __shared__ float shared[blockSize];
    shared[tid] = src[offset];
    __syncthreads();

    int off = 1;
    for (int i = blockSize / 2; i > 0; i >>= 1) { 
        if (tid < i) {
            shared[tid] += shared[tid + i];
        }
        __syncthreads();
    }

    // write back
    if (tid == 0) {
        dst[blockIdx.x] = shared[0];
    }
}



int main() {
    const int N = 32;
    size_t size = N * sizeof(float);
    float *src = new float[N], *dst = new float[N];

    for (int i = 0; i < N; i++) src[i] = 2.0;

    float *dsrc, *ddst;
    cudaMalloc(&dsrc, size);
    cudaMalloc(&ddst, size);

    cudaMemcpy(dsrc, src, size, cudaMemcpyHostToDevice);
    // reduce_nonDivergent<N><<<1, N>>>(dsrc, ddst);
    reduce_bankFree<N><<<1, N>>>(dsrc, ddst);

    cudaMemcpy(dst, ddst, size, cudaMemcpyDeviceToHost);

    std::cout << dst[0] << std::endl;        
    return 0;
}