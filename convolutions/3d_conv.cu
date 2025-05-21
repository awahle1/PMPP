#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

__global__
void 3dConvKernel(float *M, float*F, float*P, int width, int height, int depth, int r){
    int outPlane = threadIdx.z + blockDim.z*blockIdx.z;
    int outRow = threadIdx.y + blockDim.y*blockIdx.y;
    int outCol = threadIdx.x + blockDim.x*blockIdx.x;

    int F_WIDTH = 2*r + 1

    PValue = 0;
    for (int fplane=0; fplane<=F_WIDTH; ++fplane){
        for (int frow=0; frow<=F_WIDTH; ++frow){
            for (inf fcol=0; fcol<=F_WIDTH; ++fcol){
                int inCol = outCol + (fcol-2);
                int inRow = outRow + (frow-2);
                int inPlane = outPlane + (fplane-2)
                if (inCol < 0 || inRow<0 || inPlane<0 || inCol > width || inRow > height || inPlane>depth){
                    continue;
                }
                
                int i_f = fplane * F_WIDTH**2 + frow*F_WIDTH + fcol;
                int i_m = inPlane*width*height + inRow*width + inCol;
                PValue += F[i_f] * M[i_m];
                
            }
        }
    }
    P[i_l] = PValue;
    
    
}