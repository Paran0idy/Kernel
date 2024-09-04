#include "kernel.h"

// cpu version of softmax
void softmax_cpu(float* input, float* output, const int M, const int N) {
    for (int m = 0; m < M; ++m) {
        float maxval = -INFINITY;
        const float* x = input + m * N;
        for (int n = 0; n < N; ++n) {
            maxval = maxval > x[n] ? maxval : x[n];
        }
        float sumval = 0.0f;
        for (int n = 0; n < N; ++n) {
            sumval += exp(x[n] - maxval);
        }
        float* y = output + m * N;
        for (int n = 0; n < N; ++n) {
            y[n] = exp(x[n] - maxval) / sumval;
        }
    }
}

// naive gpu version of softmax
// input: M x N
// output: M x N
// each thread caculate one row of output
// assume M and N are 8192

__global__ void softmax_naive(float *input, float *output, int M, int N) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid * blockDim.x + tid;

    if (idx < M) {
        float maxval = -INFINITY;
        for (int i = 0; i < N; ++i) {
            maxval = maxval > input[idx * N + i] ? maxval : input[idx * N + i];
        }
        float sumval = 0.0f;
        for (int i = 0; i < N; ++i) {
            sumval += exp(input[idx * N + i] - maxval);
        }
        for (int i = 0; i < N; ++i) {
            output[idx * N + i] = exp(input[idx * N + i] - maxval) / sumval;
        }
    }
}


// each warp caculate one row of output, use warp reduce functions

__device__ inline float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ inline float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_warp(float *input, float *output, int M, int N) {
    const int tid = threadIdx.x;

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warpsPerBlock = blockDim.x / warpSize;
    const int numWarps = gridDim.x * warpsPerBlock;
    const int idx = warpsPerBlock * blockIdx.x + warp_id;  // start row index


    for (int m = idx; m < M; m += numWarps) {
        
        const float* x = input + m * N;
        float* const y = output + m * N;
        
        float maxval = -INFINITY;
         // each lane (thread in a warp) calculate the maxval among data with indices [landId, landId + 32, laneId + 64, ...]
        for (int n = lane_id; n < N; n += warpSize) {
            maxval = maxval > x[n] ? maxval : x[n];
        }
        maxval = warpReduceMax(maxval);

        float sumval = 0.0f;
        for (int i = lane_id; i < N; i += warpSize) {
            sumval += exp(x[i] - maxval);
        }
        sumval = warpReduceSum(sumval);

        for (int n = lane_id; n < N; n += warpSize) {
            y[n] = exp(x[n] - maxval) / sumval;
        }
    }
}

// todo: online softmax


int main() {
    int M = 8192, N = 8192;
    size_t size_io = M * N * sizeof(float);

    float *input = (float *)malloc(size_io);
    float *output = (float *)malloc(size_io);
    float *output_cpu = (float *)malloc(size_io);

    for (int i = 0; i < M * N; i++) input[i] = rand() / (float)RAND_MAX;

    float *d_input, *d_output;

    cudaMalloc(&d_input, size_io);
    cudaMalloc(&d_output, size_io);

    cudaMemcpy(d_input, input, size_io, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, size_io, cudaMemcpyHostToDevice);

    // dim3 grid(1);
    // dim3 block(M);

    dim3 grid(M/1024);
    dim3 block(1024);

    Timer gpu_timer;
    // softmax_naive<<<grid, block>>>(d_input, d_output, M, N);
    softmax_warp<<<grid, block>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();
    double gpu_time = gpu_timer.elapsed();

    Timer cpu_timer;
    softmax_cpu(input, output_cpu, M, N);
    double cpu_time = cpu_timer.elapsed();


    cudaMemcpy(output, d_output, size_io, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M * N; i++) {
        if (fabs(output[i] - output_cpu[i]) > 1e-3) {
            std::cout << "Error: " << i << " " << output[i] << " " << output_cpu[i] << std::endl;
            break;
        }
    }

    std::cout << "GPU time: " << gpu_time * 1000 << " ms" << std::endl;
    std::cout << "CPU time: " << cpu_time * 1000 << " ms" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    free(input);

    return 0;


}