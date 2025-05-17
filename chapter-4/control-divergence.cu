#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>


__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        if (i%2 == 0){
            C[i] = A[i] + B[i];
        }
        else{
            C[i] = A[i] - B[i];
        }
        
    }
}

__global__
void optimalVecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        if (i < n>>1){
            C[i*2] = A[i*2] + B[i*2];
        }
        else{
            C[2*i - n + 1] = A[2*i - n + 1] - B[2*i - n + 1];
        }
        
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n* sizeof(float);

    float *A_d;
    cudaMalloc((void**)&A_d, size);
    float *B_d;
    cudaMalloc((void**)&B_d, size);
    float *C_d;
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;

    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel 1
    cudaEventRecord(start);
    optimalVecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
    cudaEventRecord(stop);

    // Wait for kernel to finish
    cudaEventSynchronize(stop);

    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start, stop);

    printf("Kernel1 time: %.3f ms\n", milliseconds1);

    

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(C_d);
    cudaFree(B_d);
    cudaFree(A_d);

}

int main() {
    int N = 1 << 25; // 1 million elements
    size_t size = N * sizeof(float);

    // Host vectors
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    vecAdd(h_A, h_B, h_C, N);


    free(h_A); free(h_B); free(h_C);

    return 0;

}