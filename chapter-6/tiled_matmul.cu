#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

#define TILE_WIDTH 16

__global__
void matMulTiled(float* M, float* N, float* P, int Width) {
    __shared__ float MDs[TILE_WIDTH][TILE_WIDTH];
    __shared__ float NDs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = blockIdx.y*TILE_WIDTH+ threadIdx.y;
    int col = blockIdx.x*TILE_WIDTH+ threadIdx.x;

    int pvalue = 0;

    for (int ph =0; ph<Width/TILE_WIDTH; ++ph) {
        MDs[ty][tx] = M[row*Width + ph*TILE_WIDTH + tx];
        NDs[ty][tx] = N[Width*(ty + TILE_WIDTH*ph) + col];
        __syncthreads();

        for (int k=0; k<TILE_WIDTH; ++k) {
            pvalue += MDs[ty][k]*NDs[k][tx];
        }

        __syncthreads();
    }

    P[row*Width + col] = pvalue;
}

int main() {
    const int N = 1024; // Size of NxN matrix
    const size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N; ++i){
        for (int j=0; j<N; ++j){
            h_A[i*N + j] = i;
            h_B[i*N + j] = i*2;
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matMulTiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    std::cout << "C[0][0] = " << h_C[N*N-1] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}