// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

static cublasHandle_t g_handle = nullptr;

namespace py = pybind11;

std::pair<double, py::array_t<float>> cublas_dot_gpu(
    py::array_t<float> A,
    py::array_t<float> B
)
{
    const auto rows_A = A.shape(0);
    const auto cols_A = A.shape(1);
    const auto rows_B = B.shape(0);
    const auto cols_B = B.shape(1);

    if (A.ndim() != 2 || B.ndim() != 2)
        throw std::runtime_error("Expected 2D arrays");
    if (cols_A != rows_B)
        throw std::runtime_error("A and B are not compatible");

    if (A.ndim() != 2 || B.ndim() != 2)
        throw std::runtime_error("Expected 2D arrays");
    if (cols_A != rows_B)
        throw std::runtime_error("A and B are not compatible");

    const auto rows_C = rows_A;
    const auto cols_C = cols_B;
    // Création de la matrice de sortie C pour NumPy
    py::array_t<float> C({rows_C, cols_C});

    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();
    py::buffer_info bufC = C.request();

    // Mappage direct des tampons NumPy vers des vues Armadillo (zéro copie)
    // Note: Armadillo utilise par défaut le format column-major.
    // Pour du row-major, on utilise arma::fmat avec un constructeur de vue si besoin,
    // ou on mappe en tenant compte de la disposition mémoire.
    const float* ptr_A = static_cast<const float*>(bufA.ptr);
    const float* ptr_B = static_cast<const float*>(bufB.ptr);
    float* ptr_C = static_cast<float*>(bufC.ptr);

    if (!g_handle) cublasCreate(&g_handle);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(float) * rows_A * cols_A);
    cudaMalloc(&dB, sizeof(float) * rows_B * cols_B);
    cudaMalloc(&dC, sizeof(float) * rows_A * cols_B);
    cudaMemcpy(dA, const_cast<float*>(ptr_A), sizeof(float) * rows_A * cols_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, const_cast<float*>(ptr_B), sizeof(float) * rows_B * cols_B, cudaMemcpyHostToDevice);

    const float alpha = 1.f, beta = 0.f;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    cudaEventRecord(t0);

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

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float duration_ms = 0.f;
    cudaEventElapsedTime(&duration_ms, t0, t1);

    cudaMemcpy(ptr_C, dC, sizeof(float) * rows_A * cols_B, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

//    if (g_handle) { cublasDestroy(g_handle); g_handle = nullptr; }

    return {duration_ms, C};
}
