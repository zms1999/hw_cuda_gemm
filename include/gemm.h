// studentid: 2018013425
// 翟明书 计82
#include "util.h"

#define BLOCK_SIZE 32

template<class T>
__global__ void mygemm(T *A, T *B, T *C, int m, int n, int k, T alpha, T beta) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ T local_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T local_B[BLOCK_SIZE][BLOCK_SIZE];

    if( (col < n) && (row < m) )
    {
        T tmp = 0;
        
        int step = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for(int i = 0;i < step; ++i) {
            local_A[threadIdx.y][threadIdx.x] = A[row * k + i * BLOCK_SIZE + threadIdx.x];
            local_B[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * n + col];

            __syncthreads();

            for(int l = 0; l < BLOCK_SIZE; ++l)
                tmp += local_A[threadIdx.y][l] * local_B[l][threadIdx.x];

            __syncthreads();
        }
        
        C[row * n + col] = tmp * alpha + beta * C[row * n + col];
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
        dim3 grid( (N + block.x - 1) / block.x, (M + block.y - 1) / block.y );
        timestamp(t0);
//        mygemm <<<grid, block, (DIM_THREAD_BLOCK_X + DIM_THREAD_BLOCK_Y) * K * sizeof(T)>>>
        mygemm <<<grid, block>>>
            (A, B, C, M, N, K, alpha, beta);

		checkCudaErrors(cudaDeviceSynchronize());
		return 0.f;	
	}
	
}
