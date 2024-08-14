#include "kernel.h"

#define OFFSET(i, j, N) (i) * (N) + (j)
__global__ void naive_kernel(float *a, float *b, float *c, int M, int N, int K) {
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;

    if (ty < M && tx < N) {
        for (int i = 0; i < K; i++)
            c[OFFSET(ty, tx, N)] += a[OFFSET(ty, i, K)] * b[OFFSET(i, tx, N)];
    }
}

#define OFFSET(i, j, N) (i) * (N) + (j)
#define FLOAT4(pointer) reinterpret_cast<float4*> (&pointer)[0]
__global__ void threadTiling_kernel(float *a, float *b ,float *c, int M, int N, int K) {
    const int BM = 128, BN = 128, BK = 8, TILE = 8;
    int row = blockIdx.y * BM;
    int col = blockIdx.x * BN;

    __shared__ float sa[BM][BK], sb[BK][BN];
    float result[TILE][TILE] = {0};

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int smem_a_m = tid / 2;
    int smem_a_k = (tid % 2) << 2;

    int smem_b_k = tid / 32;
    int smem_b_n = (tid % 32) << 2;

    int gmem_a_m = row + smem_a_m;
    int gmem_b_n = col + smem_b_n;

    for (int k = 0; k < K / BK; k++) {
        int gmem_a_k = k * BK;
        int gmem_b_k = k * BK;

        // G2S
        FLOAT4(sa[smem_a_m][smem_a_k]) = FLOAT4(a[OFFSET(gmem_a_m, gmem_a_k, K)]);
        FLOAT4(sb[smem_b_k][smem_b_n]) = FLOAT4(b[OFFSET(gmem_b_k, gmem_b_n, N)]);
        __syncthreads();

        // compute
        int ty = threadIdx.y * TILE;
        int tx = threadIdx.x * TILE;
        for (int kk = 0; kk < BK; kk++) 
            for (int i = 0; i < TILE; i++)
                for (int j = 0; j < TILE; j++)
                    result[i][j] += sa[ty + i][kk] * sb[kk][tx + j];
        __syncthreads();
    }

    // S2G
    int ty = row + threadIdx.y * TILE;
    int tx = col + threadIdx.x * TILE;
    for (int i = 0; i < TILE; i++)
        for (int j = 0; j < TILE; j += 4)
            FLOAT4(c[OFFSET(ty + i, tx + j, N)]) = FLOAT4(result[i][j]);
}
void naive() {
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
    naive_kernel<<<block, thread>>>(da, db, dc, M, N, K);

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
}

void threadTiling() {
    const int M = 128, N = 128, K = 128;
    const int BM = 128, BN = 128, BK = 8, TILE = 8;

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
    dim3 thread(BN / TILE, BM / TILE);
    threadTiling_kernel<<<block, thread>>>(da, db, dc, M, N, K);

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
}





int main() {
    naive();
    threadTiling();

    return 0;
}