#include "kernel.h"

#define OFFSET(i, j, N) (i) * (N) + (j)
__global__ void matmul_naive(float *a, float *b, float *c, int M, int N, int K) {
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ty < M && tx < N) {
        for (int i = 0; i < K; i++)
            c[OFFSET(ty, tx, N)] += a[OFFSET(ty, i, K)] * b[OFFSET(i, tx, N)];
    }
}

int main() {
    const int M = 32, N = 32, K = 32;
    const int BM = 32, BN = 32, BK = 32;

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *a = (float *)malloc(size_a);
    float *b = (float *)malloc(size_b);
    float *c = (float *)malloc(size_c);

    for (int i = 0; i < M * K; i++) a[i] = 1.0;
    for (int i = 0; i < K * N; i++) b[i] = 1.0;

    float *da, *db, *dc;
    cudaMalloc(&da, size_a);
    cudaMalloc(&db, size_b);
    cudaMalloc(&dc, size_c);

    cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice);

    dim3 block(N / BN, M / BM);
    dim3 thread(BN, BM);
    matmul_naive<<<block, thread>>>(da, db, dc, M, N, K);

    cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            std::cout << c[OFFSET(i, j, N)] << " ";
        std::cout << std::endl;
    }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a);
    free(b);
    free(c);

    return 0;
}