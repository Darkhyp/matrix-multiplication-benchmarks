// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

__global__
void matmulKernel(
    const float* A,
    const float* B,
    float* C,
    const int M,
    const int K,
    const int N
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N)
    {
        float sum = 0.0f;

        const float* Arow = A + row * K;
        const float* Bcol = B + col;

        for (int i = 0; i < K; ++i) {
            sum += *(Arow++) * *Bcol;
            Bcol += N;
        }

        C[row * N + col] = sum;
    }
}

void launch_matmul_kernel(
    const float* dA,
    const float* dB,
    float* dC,
    const int rows_A,
    const int cols_A,
    const int cols_B,
    int blockx,
    int blocky
)
{
    dim3 block(blockx, blocky);
    dim3 grid((cols_B + block.x - 1) / block.x, (rows_A + block.y - 1) / block.y);

    // ONLY execute the triple brackets inside a .cu file compiled by NVCC!
    matmulKernel<<<grid, block>>>(dA, dB, dC, rows_A, cols_A, cols_B);
}
