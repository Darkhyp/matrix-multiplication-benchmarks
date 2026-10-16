// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

#include "../CUDA/cuda_wrapper.h"
#include <cublas_v2.h>

static cublasHandle_t g_handle = nullptr;

void launch_cublas_dot_kernel(
    const float* dA,
    const float* dB,
    float* dC,
    const int& rows_A,
    const int& cols_A,
    const int& cols_B,
    int blockx,
    int blocky
)
{
    const float alpha = 1.f, beta = 0.f;

    // TRICK ROW-MAJOR POUR CUBLAS :
    // On calcule B * A en inversant les dimensions pour s'adapter au Column-Major de cuBLAS
    cublasSgemm(
        g_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        cols_B, rows_A, cols_A,  // m = cols_B, n = rows_A, k = cols_A
        &alpha,
        dB, cols_B,              // Matrice B en premier, ldb = cols_B
        dA, cols_A,              // Matrice A en second,  lda = cols_A
        &beta,
        dC, cols_B               // Matrice C résultat,  ldc = cols_B
    );
}

// Utilise le noyau standard
std::pair<double, py::array_t<float>> cublas_dot_gpu(
    py::array_t<float> A,
    py::array_t<float> B
)
{
    if (!g_handle) cublasCreate(&g_handle);
    return cuda_dot_T<launch_cublas_dot_kernel>(A, B, 16, 16);
//    auto ret cuda_dot_T<launch_cublas_dot_kernel>(A, B, blockx, blocky);
//    if (g_handle) { cublasDestroy(g_handle); g_handle = nullptr; }
//    return ret;
}