#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>


__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
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

    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(C_d);
    cudaFree(B_d);
    cudaFree(A_d);

}

int main() {
    int N = 1 << 20; // 1 million elements
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

    std::cout << "C[0] = " << h_C[0] << ", C[N-1] = " << h_C[N-1] << std::endl;


    free(h_A); free(h_B); free(h_C);

    return 0;

}