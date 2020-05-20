// studentid: 2016123456
#include "util.h"

template<class T>
__global__ void gemm(T *A, T *B, T *C, int m, int n, int k, T alpha, T beta) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("col %d row %d\n", col, row);
    if( (col < n) && (row < m) )
    {
        T tmp = beta * C[row * n + col];
        for(int i = 0; i < k; ++i)
        {
            tmp += alpha * A[row * k + i] * B[col + i * n];
        }
        C[row * n + col] = tmp;
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
        dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
        dim3 grid( (N + block.x - 1) / block.x, (M + block.y - 1) / block.y );
        timestamp(t0);
        gemm <<<grid, block>>>
            (A, B, C, M, N, K, alpha, beta);

		checkCudaErrors(cudaDeviceSynchronize());
		return 0.f;	
	}
	
}
