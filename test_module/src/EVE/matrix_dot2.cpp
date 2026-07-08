// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

 #include <eve/module/core.hpp>
#include <vector>
#include <chrono>
#include <iostream>
#include <pybind11/numpy.h>

namespace py = pybind11;


// Computes Matrix Multiplication: C = A x B
// For simplicity and maximum SIMD efficiency, we assume standard row-major layout.
std::pair<double, py::array_t<float>> eve_dot2(
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

    // Zero the output buffer before accumulating into it
    std::fill(ptr_C, ptr_C + static_cast<size_t>(rows_C) * cols_C, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();

    const int simd_width = static_cast<int>(eve::wide<float>::size());

    // Reordered loops for streaming cache friendliness (I-K-J loop, but let's optimize pointer offsets)
    for (int i = 0; i < rows_A; ++i) {
        int i_offset_C = i * cols_C;
        int i_offset_A = i * cols_A;

        for (int k = 0; k < cols_A; ++k) {
            float a_ik = ptr_A[i_offset_A + k];
            auto wide_a = eve::wide<float>(a_ik);

            int k_offset_B = k * cols_B;
            int j = 0;

            // By utilizing direct pointers, we help MSVC optimize register allocations
            const float* b_ptr = ptr_B + k_offset_B;
            float* c_ptr = ptr_C + i_offset_C;

            for (; j <= cols_B - simd_width; j += eve::wide<float>::size()) {
                auto wide_b = eve::load(&b_ptr[j]);
                auto wide_c = eve::load(&c_ptr[j]);

                wide_c = eve::fma(wide_a, wide_b, wide_c);

                eve::store(wide_c, c_ptr + j);
            }

            for (; j < cols_B; ++j) {
                c_ptr[j] += a_ik * b_ptr[j];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    return {duration.count(), C};
}

