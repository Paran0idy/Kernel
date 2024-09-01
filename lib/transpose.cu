#include "kernel.h"

// native

__global__ void transpose_native(float* input, float* output, int M, int N) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

// memory coalescing
__global__ void transpose_mc(float* input, float* output, int M, int N) {
    const int BM = 16;
    const int BN = 16;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float tile[BM][BN];

    // g2s
    int row = by * BM + ty;
    int col = bx * BN + tx;
    if (row < M && col < N) {
        tile[ty][tx] = input[row * N + col];
    }

    __syncthreads();

    // write back to global memory, do transpose
    col = bx * BN + tx;
    row = by * BM + ty;
    if (row < M && col < N) {
        output[row * M + col] = tile[tx][ty];
    }
}

// memory coalescing wtih bank free
__global__ void transpose_mc_bank_free(float* input, float* output, int M, int N) {
    const int BM = 16;
    const int BN = 16;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float tile[BM][BN+1];

    // g2s
    int row = by * BM + ty;
    int col = bx * BN + tx;
    if (row < M && col < N) {
        tile[ty][tx] = input[row * N + col];
    }

    __syncthreads();

    // write back to global memory, do transpose
    col = bx * BN + tx;
    row = by * BM + ty;
    if (row < M && col < N) {
        output[row * M + col] = tile[tx][ty];
    }
}



template<const int M, const int N, void (*F)(float*, float*, int, int)>
void call_kernel() {
    float* input = (float*)malloc(M * N * sizeof(float));
    float* output = (float*)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * N; i++) {
        input[i] = i;
    }

    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, M * N * sizeof(float));
    cudaMalloc(&d_output, M * N * sizeof(float));

    cudaMemcpy(d_input, input, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    F<<<grid, block>>>(d_input, d_output, M, N);
    cudaEventRecord(stop);

    cudaMemcpy(output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    // caculate bandwidth
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float bandwidth = (M * N * sizeof(float) * 2) / milliseconds / 1e6;
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;


    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < M; j++) {
    //         std::cout << output[i * M + j] << " ";
    //         if (j == M - 1) std::cout << std::endl;
    //     }
    // }

    free(input);
    free(output);

    cudaFree(d_input);
    cudaFree(d_output);
}






int main() {

    call_kernel<20000, 15000, transpose_native>();
    call_kernel<20000, 15000, transpose_mc>();
    call_kernel<20000, 15000, transpose_mc_bank_free>();

    return 0;
}