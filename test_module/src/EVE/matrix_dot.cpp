// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

#include <eve/module/core.hpp>
#include <vector>
#include <chrono>
#include <iostream>
#include <thread>
#include <omp.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Computes Matrix Multiplication: C = A x B
// For simplicity and maximum SIMD efficiency, we assume standard row-major layout.
std::pair<double, py::array_t<float>> eve_dot(
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

    auto bufA = A.request();
    auto bufB = B.request();

    const auto rows_C = rows_A;
    const auto cols_C = cols_B;
    py::array_t<float> C({rows_C, cols_C});
    auto bufC = C.request();

    const float* ptr_A = static_cast<const float*>(bufA.ptr);
    const float* ptr_B = static_cast<const float*>(bufB.ptr);
    float* ptr_C = static_cast<float*>(bufC.ptr);

    // Zero the output buffer before accumulating into it
    std::fill(ptr_C, ptr_C + static_cast<size_t>(rows_C) * cols_C, 0.0f);

    // Détection dynamique du nombre de cœurs/threads disponibles
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4; // Valeur de secours si la détection échoue
    }
    omp_set_num_threads(num_threads);
//    std::cout << "Threads utilisés : " << omp_get_max_threads() << std::endl;

    // Start timing
    auto start = std::chrono::steady_clock::now();

    const auto simd_width = eve::wide<float>::size();

    // Perform Matrix Multiplication using EVE for the innermost loop
    // schedule(guided) ou schedule(dynamic) peut être utile si la charge par ligne varie
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < rows_A; ++i) {
        for (std::ptrdiff_t k = 0; k < cols_A; ++k) {
            float a_ik = ptr_A[i * cols_A + k];

            // Broadcast a_ik into an EVE wide register
            auto wide_a = eve::wide<float>(a_ik);

            std::ptrdiff_t j = 0;
            // Vectorized loop over columns of B / C
            for (; j <= cols_B - simd_width; j += eve::wide<float>::size()) {
                auto wide_b = eve::load(ptr_B + k * cols_B + j);
                auto wide_c = eve::load(ptr_C + i * cols_C + j);

                // Fused Multiply-Add: c = c + (a * b)
                wide_c = eve::fma(wide_a, wide_b, wide_c);

                eve::store(wide_c, ptr_C + i * cols_C + j);
            }

            // Scalar tail cleanup for remaining columns
            for (; j < cols_B; ++j) {
                ptr_C[i * cols_C + j] += a_ik * ptr_B[k * cols_B + j];
            }
        }
    }

    // End timing
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    return {duration.count(), C};
}