// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

#include <Kokkos_Core.hpp>
#include <chrono>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// #define CHRONO

namespace py = pybind11;

// Device view remains layout-agnostic (defaults to LayoutLeft on CUDA for maximum performance)
using DeviceView2D = Kokkos::View<float**, Kokkos::LayoutRight>;

// Explicitly define a Host View that matches NumPy's standard Row-Major layout
using HostView2D =
    Kokkos::View<
        float**,
        Kokkos::LayoutRight,
        Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>
    >;
using ConstHostView2D =
    Kokkos::View<
        const float**,
        Kokkos::LayoutRight,
        Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using TeamPolicy     = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecutionSpace>;

std::pair<double, py::array_t<float>> kokkos_dot(
    py::array_t<float> A,
    py::array_t<float> B,
    int blockx,
    int blocky
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

    if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
    }

    py::array_t<float> C({rows_A, cols_B});

    // Get raw pointers directly from NumPy array buffers
    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();
    py::buffer_info bufC = C.request();

    const float* ptr_A = static_cast<const float*>(bufA.ptr);
    const float* ptr_B = static_cast<const float*>(bufB.ptr);
    float* ptr_C = static_cast<float*>(bufC.ptr);

    // Allocate optimized layout views on the GPU Device Space
    ConstHostView2D host_A(ptr_A, rows_A, cols_A);
    ConstHostView2D host_B(ptr_B, cols_A, cols_B);
    HostView2D host_C(ptr_C, rows_A, cols_B);

    DeviceView2D dev_A("dev_A", rows_A, cols_A);
    DeviceView2D dev_B("dev_B", cols_A, cols_B);
    DeviceView2D dev_C("dev_C", rows_A, cols_B);

    // Deep Copy to GPU VRAM.
    // Kokkos automatically converts LayoutRight (Host) -> LayoutLeft (Device) behind the scenes.
    Kokkos::deep_copy(dev_A, host_A);
    Kokkos::deep_copy(dev_B, host_B);

    // --- Pure Execution Benchmark ---
    double elapsed_ms = 0.0;
#ifdef CHRONO
    auto start = std::chrono::steady_clock::now();
#else
    Kokkos::Timer timer;
#endif

    // Execute matrix multiplication using optimized 16x16 CUDA Thread Blocks
    TeamPolicy policy({0, 0}, {rows_A, cols_B}, {blockx, blocky});

    Kokkos::parallel_for("matmul", policy,
    KOKKOS_LAMBDA (const int i, const int j) {
        float sum = 0.0f;
        for (int k = 0; k < cols_A; ++k) {
            sum += dev_A(i, k) * dev_B(k, j);
        }
        dev_C(i, j) = sum;
    });

    // Sync and block host execution thread until GPU finishes the kernel pipeline
    Kokkos::fence();

#ifdef CHRONO
    auto end = std::chrono::steady_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
#else
    elapsed_ms = timer.seconds() * 1000.0;
#endif

    Kokkos::deep_copy(host_C, dev_C);

    return {elapsed_ms, C};
}

void finalize_kokkos_once() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized()) {
        Kokkos::fence();
        Kokkos::finalize();
    }
}
