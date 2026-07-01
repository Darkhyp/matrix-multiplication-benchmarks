__global__
void matmulKernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N)
    {
        float sum = 0.0f;

        for(int i=0;i<K;i++)
            sum += A[row*K+i] * B[i*N+col];

        C[row*N+col] = sum;
    }
}

void launch_matmul_kernel(const float* dA, const float* dB, float* dC, int rows_A, int cols_A, int cols_B) {
    dim3 block(16, 16);
    dim3 grid((cols_B + block.x - 1) / block.x, (rows_A + block.y - 1) / block.y);

    // ONLY execute the triple brackets inside a .cu file compiled by NVCC!
    matmulKernel<<<grid, block>>>(dA, dB, dC, rows_A, cols_A, cols_B);

    // Check for launch errors
    cudaDeviceSynchronize();
}
