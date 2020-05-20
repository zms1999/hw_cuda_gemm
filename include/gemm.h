// studentid: 2018013425
// 翟明书 计82
#include "util.h"
#include <algorithm>

#define BLOCK_SIZE 8

template<class T>
__global__ void mygemm(T *A, T *B, T *C, int m, int n, int k, T alpha, T beta) {
    int col = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    int row = blockIdx.y * blockDim.y * 4 + threadIdx.y;

    __shared__ T local_A[BLOCK_SIZE*4][BLOCK_SIZE*4];        
    __shared__ T local_B[BLOCK_SIZE*4][BLOCK_SIZE*4];


    if( !((col >= n) && (row >= m)) )
    {
        T tmp[4][4] = {static_cast<T>(0.0)};
        T reg_A[4],reg_B[4];
        
        int step = (k + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);
        for(int i = 0;i < step - 1; ++i) {
            #pragma unroll
            for(int s1 = 0; s1 < 4; ++s1)
                #pragma unroll
                for(int s2 = 0; s2 < 4; ++ s2) {
                    local_A[s1 * BLOCK_SIZE + threadIdx.y][s2 * BLOCK_SIZE + threadIdx.x] = A[(row + s1 * BLOCK_SIZE) * k + threadIdx.x + i * BLOCK_SIZE * 4 + BLOCK_SIZE * s2];
                    local_B[s1 * BLOCK_SIZE + threadIdx.y][s2 * BLOCK_SIZE + threadIdx.x] = B[(i * BLOCK_SIZE * 4 + BLOCK_SIZE * s1 + threadIdx.y) * n + col + BLOCK_SIZE * s2];
                }
            
            __syncthreads();

            if(col < n && row < m)
                #pragma unroll
                for(int l = 0; l < BLOCK_SIZE * 4; ++l) {
                    
                    #pragma unroll
                    for(int s1 = 0; s1 < 4; ++s1) {
                        reg_A[s1] = local_A[threadIdx.y + s1 * BLOCK_SIZE][l];
                        reg_B[s1] = local_B[l][threadIdx.x + s1 * BLOCK_SIZE];
                    }
                    
                    #pragma unroll
                    for(int s1 = 0; s1 < 4; ++s1)
                        #pragma unroll
                        for(int s2 = 0; s2 < 4; ++s2) {
                            tmp[s1][s2] += reg_A[s1] * reg_B[s2];
                        }

                }

            __syncthreads();
        }
        
        int i = step - 1;
        #pragma unroll
        for(int s1 = 0; s1 < 4; ++s1)
            #pragma unroll
            for(int s2 = 0; s2 < 4; ++ s2) {
                local_A[s1 * BLOCK_SIZE + threadIdx.y][s2 * BLOCK_SIZE + threadIdx.x] = A[(row + s1 * BLOCK_SIZE) * k + threadIdx.x + i * BLOCK_SIZE * 4 + BLOCK_SIZE * s2];
                local_B[s1 * BLOCK_SIZE + threadIdx.y][s2 * BLOCK_SIZE + threadIdx.x] = B[(i * BLOCK_SIZE * 4 + BLOCK_SIZE * s1 + threadIdx.y) * n + col + BLOCK_SIZE * s2];
            }
        int kmod = k % (BLOCK_SIZE * 4);
        int len = BLOCK_SIZE * 4;
        if(kmod) len = kmod;
        
        __syncthreads();

        
        if(col < n && row < m)
            for(int l = 0; l < len; ++l) {

                #pragma unroll
                for(int s1 = 0; s1 < 4; ++s1) {
                    reg_A[s1] = local_A[threadIdx.y + s1 * BLOCK_SIZE][l];
                    reg_B[s1] = local_B[l][threadIdx.x + s1 * BLOCK_SIZE];
                }

                #pragma unroll
                for(int s1 = 0; s1 < 4; ++s1)
                    #pragma unroll
                    for(int s2 = 0; s2 < 4; ++s2) {
                        tmp[s1][s2] += reg_A[s1] * reg_B[s2];
                    }

            }
        if(col + BLOCK_SIZE * 3 < n && row + BLOCK_SIZE * 3 < m)
            #pragma unroll
            for(int s1 = 0; s1 < 4; ++s1)
                #pragma unroll
                for(int s2 = 0; s2 < 4; ++s2)
                    C[(row + BLOCK_SIZE * s1) * n + col + BLOCK_SIZE * s2] = tmp[s1][s2] * alpha + beta * C[(row + BLOCK_SIZE * s1) * n + col + BLOCK_SIZE * s2];
        else 
            #pragma unroll
            for(int s1 = 0; s1 < 4; ++s1)
                #pragma unroll
                for(int s2 = 0; s2 < 4; ++s2)
                    if( col + BLOCK_SIZE * s2 < n && row + BLOCK_SIZE * s1 < m)
                        C[(row + BLOCK_SIZE * s1) * n + col + BLOCK_SIZE * s2] = tmp[s1][s2] * alpha + beta * C[(row + BLOCK_SIZE * s1) * n + col + BLOCK_SIZE * s2];
    }
}

    template<class T>
double myGEMM(T* A, T* B, T* C, T alpha, T beta)
{
    printf("perform your gemm here on m=%d n=%d k=%d\n", M, N, K);
    bool preprocess = false;
    if(preprocess)
    {
        // your preprocess
        timestamp(t0);
        // your gemm

        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t1);
        return getDuration(t0, t1);
    }
    else
    {
        // your gemm
        dim3 block(BLOCK_SIZE,BLOCK_SIZE);
        dim3 grid( (N + block.x * 4 - 1) / (block.x * 4), (M + block.y * 4- 1) / (block.y * 4) );
        timestamp(t0);
        //        mygemm <<<grid, block, (DIM_THREAD_BLOCK_X + DIM_THREAD_BLOCK_Y) * K * sizeof(T)>>>
        mygemm <<<grid, block>>>
            (A, B, C, M, N, K, alpha, beta);

        checkCudaErrors(cudaDeviceSynchronize());
        return 0.f;	
    }

}
