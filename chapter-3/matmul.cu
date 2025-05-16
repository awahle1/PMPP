#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>


__global__
void matMulRowKernel(float* M, float* N, float* P, int m1, int n1, int m2, int n2) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;

    for(int k=0; k<n2; ++k){
        int val = 0;
        for(int l=0; l<n1; ++l){
            val += M[row*n1 + l] *  N[k+l*n2];
        }
        P[row*n2 + k] = val;
    }
}

__global__
void matMulColKernel(float* M, float* N, float* P, int m1, int n1, int m2, int n2) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;

    for(int k=0; k<m1; ++k){
        int val = 0;
        for(int l=0; l<m2; ++l){
            val += M[k*n1 + l] *N[col+l*n2];
        }
        P[row*n1 + col] = val;
    }
}

void matMulRow(float* M_h, float* N_h, float* P_h, int m1, int n1, int m2, int n2) {

    float *M_d;
    cudaMalloc((void**)&M_d, (m1*n1)*sizeof(float));
    float *N_d;
    cudaMalloc((void**)&N_d, (m2*n2)*sizeof(float));
    float *Pr_d;
    cudaMalloc((void**)&Pr_d, (m1*n2)*sizeof(float));

    cudaMemcpy(M_d, M_h, (m1*n1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, (m2*n2)*sizeof(float), cudaMemcpyHostToDevice);

    int blockHeight = 2;

    dim3 dimBlock(ceil(m1/blockHeight), 1, 1);
    dim3 dimThread(blockHeight, 1, 1);

    printf("dimBlock: x=%u, y=%u, z=%u\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf("dimThread: x=%u, y=%u, z=%u\n", dimThread.x, dimThread.y, dimThread.z);

    matMulRowKernel<<<dimBlock, dimThread>>>(M_d, N_d, Pr_d, m1, n1, m2, n2);
    cudaMemcpy(P_h, Pr_d, (m1*n2)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(Pr_d);

    cudaFree(M_d);
    cudaFree(N_d);

}

void matMulCol(float* M_h, float* N_h, float* P_h, int m1, int n1, int m2, int n2) {

    float *M_d;
    cudaMalloc((void**)&M_d, m1*n1);
    float *N_d;
    cudaMalloc((void**)&N_d, m2*n2);
    float *Pc_d;
    cudaMalloc((void**)&Pc_d, m1*n2);

    cudaMemcpy(M_d, M_h, m1*n1, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, m2*n2, cudaMemcpyHostToDevice);

    int blockWidth = 2;
    dim3 dimBlockCol(1, ceil(n2/blockWidth), 1);;
    dim3 dimThreadCol(1, blockWidth, 1);

    matMulColKernel<<<dimBlockCol, dimThreadCol>>>(M_d, N_d, Pc_d, m1, n1, m2, n2);
    cudaMemcpy(P_h, Pc_d, m1*n2, cudaMemcpyDeviceToHost);
    cudaFree(Pc_d);
    
    cudaFree(M_d);
    cudaFree(N_d);

}

int main() {
    // Matrix dimensions
    int m1 = 2, n1 = 3;
    int m2 = 3, n2 = 2;

    if (n1 != m2) {
        fprintf(stderr, "Matrix dimensions are incompatible for multiplication\n");
        return 1;
    }

    // Allocate and initialize M_h (2x3)
    float M_h[6] = {
        1, 2, 3,
        4, 5, 6
    };

    // Allocate and initialize N_h (3x2)
    float N_h[6] = {
        7, 8,
        9, 10,
        11, 12
    };

    // Allocate result matrix P_h (2x2)
    float P_h[4] = {0};

    // Call matMul function
    matMulRow(M_h, N_h, P_h, m1, n1, m2, n2);

    // Print result
    printf("Result matrix P (2x2):\n");
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            printf("%f ", P_h[i * n2 + j]);
        }
        printf("\n");
    }

    return 0;
}