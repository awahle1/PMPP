void vecAdd(float* A, float* B, float* C, int n) {
    int size = n* sizeof(float);

    float *A_d;
    cudaMalloc((void**)&A_d, size);
    float *B_d;
    cudaMalloc((void**)&B_d, size);
    float *C_d;
    cudaMalloc((void**)^C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, A_h, size, cudaMemcpyHostToDevice);

    

    cudaMemcpy(C_h, C_d, size, cudeMemcpyDeviceToHost);

    cudaFree(C_d);
    cudaFree(B_d);
    cudaFree(A_d);

}

int main() {

}