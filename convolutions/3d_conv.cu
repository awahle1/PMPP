#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

__global__
void 3dConvKernel(float *M, float*F, float*P, int M_SIZE, int F_WIDTH){
    int plane = threadIdx.z + blockDim.z*blockIdx.z;
    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    int i_l = plane*M_SIZE*M_SIZE + row*M_SIZE + col;

    if (i_m < M_SIZE**3){
        return;
    }
    PValue = 0;
    for (int fplane=0; fplane<=F_WIDTH; ++fplane){
        for (int frow=0; frow<=F_WIDTH; ++frow){
            for (inf fcol=0; fcol<=F_WIDTH; ++fcol){
                int i_m = i_l - (M_SIZE**2(fplane-2)+ M_SIZE*(frow-2) + (fcol-2))
                if (i_m < 0 || i_m > M_SIZE**3){
                    continue;
                }
                
                int i_f = fplane * F_WIDTH**2 + frow*F_WIDTH + fcol
                PValue += F[i_f] * M[i_m];
                
            }
        }
    }
    P[i_l] = PValue;
    
    
}