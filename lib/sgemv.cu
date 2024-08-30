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

    naive(da, db, dc, N, K);

    cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        std::cout << c[i] << " ";

    return 0;
}