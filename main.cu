#include <iostream>
#include "input.hpp"
#include "debug.cu"

const unsigned long long N = pow(512, 3);

__global__ void vector_nand(int *out, int *a, int *b, int n) {
    int gridSize = gridDim.x * gridDim.y * gridDim.z;
    int threadSize = blockDim.x * blockDim.y * blockDim.z;

    int blockIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int threadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    int index = blockIndex * threadSize + threadIndex;
    int stride = gridSize * threadSize;

    for (int i = index; i < n; i += stride) {
        out[i] = ~(a[i] & b[i]);
    }
}

void nand(int *a, int *b, int *out, int n) {
    int *d_a, *d_b, *d_out;

    cudaMalloc((void**)&d_a, sizeof(int) * n);
    cudaMalloc((void**)&d_b, sizeof(int) * n);
    cudaMalloc((void**)&d_out, sizeof(int) * n);

    cudaMemcpy(d_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);

    dim3 dimGrid(8, 8, 8);
    dim3 dimBlock(8, 8, 8);
    vector_nand<<<dimGrid, dimBlock>>>(d_out, d_a, d_b, n);

    cudaMemcpy(out, d_out, sizeof(int) * N, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

int main(){
    int *a, *b, *out;

    a = (int*)malloc(sizeof(int) * N);
    b = (int*)malloc(sizeof(int) * N);
    out = (int*)malloc(sizeof(int) * N);

    for(int i = 0; i < N; i++) {
        generate_data(&a[i], &b[i], i);
    }

    nand(a, b, out, N);

    dump(a, b, out, 64);
    test(out, N);

    free(a); 
    free(b); 
    free(out);
}
