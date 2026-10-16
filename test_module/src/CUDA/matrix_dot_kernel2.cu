// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

#define TILE_WIDTH 16

__global__
void matmulKernelOptimized(
    const float* A,
    const float* B,
    float* C,
    const int M,
    const int K,
    const int N
)
{
    // Déclaration de la mémoire partagée pour les tuiles de A et B
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    // Boucle sur toutes les tuiles nécessaires pour couvrir la dimension K
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {

        // Chargement collaboratif de la tuile de A dans la mémoire partagée
        if (row < M && (t * TILE_WIDTH + tx) < K) {
            s_A[ty][tx] = A[row * K + t * TILE_WIDTH + tx];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Chargement collaboratif de la tuile de B dans la mémoire partagée
        if (col < N && (t * TILE_WIDTH + ty) < K) {
            s_B[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        // Attendre que tous les threads du bloc aient fini de charger leurs données
        __syncthreads();

        // Calcul du produit partiel pour cette tuile
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += s_A[ty][i] * s_B[i][tx];
        }

        // Attendre avant de charger la tuile suivante pour éviter les conflits de lecture
        __syncthreads();
    }

    // Écriture du résultat final dans la mémoire globale
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_matmul_kernel_optimized(
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
    matmulKernelOptimized<<<grid, block>>>(dA, dB, dC, rows_A, cols_A, cols_B);
}
