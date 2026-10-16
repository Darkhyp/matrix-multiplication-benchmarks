// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

#pragma once

#ifdef _WIN32
#include <windows.h>
extern "C" void _gfortran_system_clock_8(long long* count, long long* rate, long long* max) {
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    if (count) *count = c.QuadPart;
    if (rate)  *rate  = f.QuadPart;
    if (max)   *max   = 0x7fffffffffffffffLL;
}
#endif

extern "C" {
    void matmul_cpu_(
        const float* A,
        const float* B,
        float* C,
        const int* N,
        const int* M,
        const int* P,
        double* elapsed_ms
    );
    void matmul_cpu_sgemm_(
        const float* A,
        const float* B,
        float* C,
        const int* N,
        const int* M,
        const int* P,
        double* elapsed_ms
    );
#ifdef HAS_GPU_FORTRAN
    void matmul_gpu_(
        const float* A,
        const float* B,
        float* C,
        const int* N,
        const int* M,
        const int* P,
        double* elapsed_ms
    );
#endif
}

template<auto LaunchFunc>
std::pair<double, py::array_t<float>> matmul_wrapper(
    py::array_t<float> A,
    py::array_t<float> B
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

    py::array_t<float, py::array::f_style> C({rows_A, cols_B});
    py::buffer_info bufC = C.request();

    const float* ptr_A = static_cast<const float*>(bufA.ptr);
    const float* ptr_B = static_cast<const float*>(bufB.ptr);
    float* ptr_C = static_cast<float*>(bufC.ptr);

    double elapsed_ms = 0.0;

    LaunchFunc(
        ptr_A,
        ptr_B,
        ptr_C,
        &rows_A,
        &cols_A,
        &cols_B,
        &elapsed_ms
    );

    return {elapsed_ms, C};
}
