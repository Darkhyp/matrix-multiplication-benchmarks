#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "matrix_dot.hu"
#include <pybind11/numpy.h>

namespace py = pybind11;


std::pair<double, py::array_t<float>> cuda_dot(
    py::array_t<float> A,
    py::array_t<float> B
)
{
    const int rows_A = A.shape(0);
    const int cols_A = A.shape(1);
    const int rows_B = B.shape(0);
    const int cols_B = B.shape(1);

    if (A.ndim() != 2 || B.ndim() != 2)
        throw std::runtime_error("Expected 2D arrays");
    if (cols_A != rows_B)
        throw std::runtime_error("A and B are not compatible");

    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();

    const int rows_C = rows_A;
    const int cols_C = cols_B;
    py::array_t<float> C({rows_C, cols_C});
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
    auto start = std::chrono::high_resolution_clock::now();

    launch_matmul_kernel(dA, dB, dC, rows_A, cols_A, cols_B);

    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    cudaMemcpy(ptr_C, dC, rows_A * cols_B * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return {duration.count(), C};
}