// 3d_conv_host.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error " << cudaGetErrorName(err) << " ("         \
                      << cudaGetErrorString(err) << ") at " << __FILE__ << ':'  \
                      << __LINE__ << std::endl;                                 \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

__global__
void conv3DKernel(float *M, float*F, float*P, int width, int height, int depth, int r){
    int outPlane = threadIdx.z + blockDim.z*blockIdx.z;
    int outRow = threadIdx.y + blockDim.y*blockIdx.y;
    int outCol = threadIdx.x + blockDim.x*blockIdx.x;

    int F_WIDTH = 2*r + 1;

    float PValue = 0;
    for (int fplane=0; fplane<F_WIDTH; ++fplane){
        for (int frow=0; frow<F_WIDTH; ++frow){
            for (int fcol=0; fcol<F_WIDTH; ++fcol){
                int inCol = outCol + (fcol-r);
                int inRow = outRow + (frow-r);
                int inPlane = outPlane + (fplane-r);
                if (inCol < 0 || inRow<0 || inPlane<0 || inCol >= width || inRow >= height || inPlane>=depth){
                    continue;
                }                
                
                int i_f = fplane * F_WIDTH*F_WIDTH + frow*F_WIDTH + fcol;
                int i_m = inPlane*width*height + inRow*width + inCol;
                PValue += F[i_f] * M[i_m];
                
            }
        }
    }
    int i_p = outPlane*width*height + outRow*width + outCol;
    if (i_p < width*height*depth){
        P[i_p] = PValue;
    }
    
}



/* ----------------------------------------------------------------------------
   Helpers
---------------------------------------------------------------------------- */
inline int idx3d(int x, int y, int z, int W, int H) {
    return z * (W * H) + y * W + x;   // row-major-within-slice, slice-major
}

void cpuReference3dConv(const std::vector<float> &M,
                        const std::vector<float> &F,
                        std::vector<float> &P,
                        int W, int H, int D, int r)
{
    const int fDim = 2 * r + 1;
    for (int z = 0; z < D; ++z)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float acc = 0.f;
                for (int kz = -r; kz <= r; ++kz)
                    for (int ky = -r; ky <= r; ++ky)
                        for (int kx = -r; kx <= r; ++kx) {
                            int zx = x + kx; int zy = y + ky; int zz = z + kz;
                            if (zx < 0 || zx >= W ||
                                zy < 0 || zy >= H ||
                                zz < 0 || zz >= D)
                                continue;                 // zero-padding
                            float m = M[idx3d(zx, zy, zz, W, H)];
                            float f = F[idx3d(kx + r, ky + r, kz + r,
                                              fDim, fDim)];
                            acc += m * f;
                        }
                P[idx3d(x, y, z, W, H)] = acc;
            }
}

bool almostEqual(float a, float b, float eps = 1e-2f) {
    return std::fabs(a - b) <= eps * std::max(std::fabs(a), std::fabs(b));
}

/* ----------------------------------------------------------------------------
   main
---------------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    /* ─── Problem size ───────────────────────────────────────────────────── */
    const int W = 64, H = 64, D = 64;   // volume dimensions
    const int r = 1;                    // filter radius → (2r+1)^3 elements
    const int volSize  = W * H * D;
    const int fSize    = (2 * r + 1) * (2 * r + 1) * (2 * r + 1);
    const size_t volBytes = volSize * sizeof(float);
    const size_t fBytes   = fSize * sizeof(float);

    std::cout << "Volume  : " << W << 'x' << H << 'x' << D << '\n'
              << "Filter  : (" << 2*r+1 << ")^3 = " << fSize << " elements\n";

    /* ─── Host buffers ───────────────────────────────────────────────────── */
    std::vector<float> h_M(volSize), h_F(fSize), h_P(volSize), ref(volSize);

    for (auto &v : h_M) v = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (auto &v : h_F) v = static_cast<float>(rand()) / RAND_MAX - 0.5f;

    /* ─── Device buffers ─────────────────────────────────────────────────── */
    float *d_M = nullptr, *d_F = nullptr, *d_P = nullptr;
    CHECK_CUDA(cudaMalloc(&d_M, volBytes));
    CHECK_CUDA(cudaMalloc(&d_F, fBytes));
    CHECK_CUDA(cudaMalloc(&d_P, volBytes));

    CHECK_CUDA(cudaMemcpy(d_M, h_M.data(), volBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_F, h_F.data(), fBytes,   cudaMemcpyHostToDevice));

    /* ─── Launch kernel ──────────────────────────────────────────────────── */
    dim3 blockDim(8, 8, 8);                           // 512 threads/block
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x,
                 (H + blockDim.y - 1) / blockDim.y,
                 (D + blockDim.z - 1) / blockDim.z);

    conv3DKernel<<<gridDim, blockDim>>>(d_M, d_F, d_P, W, H, D, r);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* ─── Copy result back ──────────────────────────────────────────────── */
    CHECK_CUDA(cudaMemcpy(h_P.data(), d_P, volBytes, cudaMemcpyDeviceToHost));

    /* ─── Verification ──────────────────────────────────────────────────── */
    std::cout << "Computing reference on CPU …" << std::flush;
    cpuReference3dConv(h_M, h_F, ref, W, H, D, r);
    std::cout << " done.\nChecking correctness … ";

    size_t mismatch = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!almostEqual(ref[i], h_P[i])) {
            ++mismatch;
            if (mismatch < 5)
                std::cerr << "Mismatch at i=" << i
                          << " ref=" << ref[i] << " gpu=" << h_P[i] << '\n';
        }
    }
    if (mismatch == 0)
        std::cout << "PASSED 🎉\n";
    else
        std::cout << "FAILED (" << mismatch << " mismatches)\n";

    /* ─── Cleanup ───────────────────────────────────────────────────────── */
    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_F));
    CHECK_CUDA(cudaFree(d_P));
    return (mismatch == 0) ? 0 : 1;
}