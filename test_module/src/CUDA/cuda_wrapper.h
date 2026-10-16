// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <utility>
#include <cuda_runtime.h>

namespace py = pybind11;

//#define DEBUG_SYNC
//#define CHRONO

#define CUDA_CHECK(call)                                                \
    {                                                                   \
        cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess)                                        \
            throw std::runtime_error(std::string(__FILE__) + ":" +      \
                std::to_string(__LINE__) + " kernel launch failed: " +  \
                cudaGetErrorString(err_));                              \
    }

// LaunchFunc = launch_matmul_kernel or launch_matmul_kernel_optimized
template<auto LaunchFunc>
std::pair<double, py::array_t<float>> cuda_dot_T(
    py::array_t<float> A,
    py::array_t<float> B,
    int blockx, int blocky
)
{
    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();

    const int rows_A = bufA.shape[0];
    const int cols_A = bufA.shape[1];
    const int rows_B = bufB.shape[0];
    const int cols_B = bufB.shape[1];

    if (A.ndim() != 2 || B.ndim() != 2)
        throw std::runtime_error("Expected 2D arrays");
    if (cols_A != rows_B)
        throw std::runtime_error("A and B are not compatible");

    py::array_t<float> C({rows_A, cols_B});
    py::buffer_info bufC = C.request();

    const float* ptr_A = static_cast<const float*>(bufA.ptr);
    const float* ptr_B = static_cast<const float*>(bufB.ptr);
    float* ptr_C = static_cast<float*>(bufC.ptr);

    // Device pointers
    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    cudaMalloc(&dA, rows_A * cols_A * sizeof(float));
    cudaMalloc(&dB, cols_A * cols_B * sizeof(float));
    cudaMalloc(&dC, rows_A * cols_B * sizeof(float));
    cudaMemcpy(dA, ptr_A, rows_A * cols_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, ptr_B, cols_A * cols_B * sizeof(float), cudaMemcpyHostToDevice);

    // Start timing
#ifdef CHRONO
    #include <chrono>
    auto start = std::chrono::steady_clock::now();
#else
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
#endif

    LaunchFunc(dA, dB, dC, rows_A, cols_A, cols_B, blockx, blocky);
    CUDA_CHECK(cudaGetLastError())

    cudaEventRecord(stop);
    // Wait until the kernel has finished
    cudaEventSynchronize(stop);
#ifdef CHRONO
    auto end = std::chrono::steady_clock::now();
#endif

#ifdef DEBUG_SYNC
    CUDA_CHECK(cudaEventSynchronize(stop))
#endif

#ifdef CHRONO
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
#else
    float duration_ms = 0.0f;
    cudaEventElapsedTime(&duration_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

    cudaMemcpy(ptr_C, dC, rows_A * cols_B * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return {duration_ms, C};
}