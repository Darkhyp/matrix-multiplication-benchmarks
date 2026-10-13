// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

#include <chrono>
#include <vector>
#include <thread>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// Include Kiwaku headers
#include <kwk/kwk.hpp>

namespace py = pybind11;

std::pair<double, py::array_t<float>> kiwaku_dot(
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

    const auto rows_C = rows_A;
    const auto cols_C = cols_B;
    py::array_t<float> C({rows_C, cols_C});

    // Get raw pointers directly from NumPy array buffers
    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();
    py::buffer_info bufC = C.request();

    const float* ptr_A = static_cast<const float*>(bufA.ptr);
    const float* ptr_B = static_cast<const float*>(bufB.ptr);
    float* ptr_C = static_cast<float*>(bufC.ptr);

    // Correct way to build views from raw pointers in Kiwaku
    auto kiwaku_A = kwk::view{ kwk::source = ptr_A, kwk::of_size(rows_A, cols_A) };
    auto kiwaku_B = kwk::view{ kwk::source = ptr_B, kwk::of_size(cols_A, cols_B) };
    auto kiwaku_C = kwk::view{ kwk::source = ptr_C, kwk::of_size(rows_A, cols_B) };

    // Configuration dynamique des cœurs/threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    omp_set_num_threads(num_threads);

    // Measure Execution performance loop
    auto start = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < cols_A; ++k) {
                sum += kiwaku_A(i, k) * kiwaku_B(k, j);
            }
            kiwaku_C(i, j) = sum;
        }
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return {elapsed_ms, C};
}
