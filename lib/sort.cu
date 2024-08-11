#include "kernel.h"

template<const int N>
__global__ void mergeSort(float *a, float *b) {
    size_t sizePerThread = N / blockDim.x;
    __shared__ float sa[N];
    __shared__ float tmp[N];

    // g2s
    sa[threadIdx.x] = a[threadIdx.x];
    __syncthreads();

    // sort
    sizePerThread *= 2;
    for (int start = sizePerThread * threadIdx.x; start + sizePerThread <= N; start *= 2) {
        size_t end = start + sizePerThread;
        sizePerThread *= 2;

        int mid = start + (end - start) / 2;
        int i = start, j = mid, k = start;
        while (i < mid && j < end) {
            if (sa[i] < sa[j]) tmp[k++] = sa[i++];
            else tmp[k++] = sa[j++];
        }

        while (i < mid) tmp[k++] = sa[i++];
        while (j < end) tmp[k++] = sa[j++];
        __syncthreads();
        
        for (int k = start; k < end; k++)
            sa[k] = tmp[k];
        __syncthreads();
    }

    for (int i = 0; i < N; i++)
        b[i] = sa[i];
}


int main() {
    const int N = 32;
    size_t size = sizeof(float) * N;

    float *a = (float *)malloc(size);
    float *c = (float *)malloc(size);

    for (int i = 0; i < N; i++) a[i] = N - i;

    float *da, *dc;

    cudaMalloc(&da, size);
    cudaMalloc(&dc, size);

    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(N);

    mergeSort<N><<<grid, block>>>(da, dc);
    cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) std::cout << c[i] << " ";
    std::cout << std::endl;

    return 0;
}