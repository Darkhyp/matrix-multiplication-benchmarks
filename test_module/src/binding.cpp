// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "CUDA/matrix_op.h"
#include "EVE/matrix_op.h"
#include "EIGEN/matrix_op.h"
#include "ARMADILLO/matrix_op.h"
#include "KOKKOS/matrix_op.h"
#include "KIWAKU/matrix_op.h"

namespace py = pybind11;

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

#ifdef HAS_FORTRAN
using matmul_func = void (*)(
    float*,
    float*,
    float*,
    int*,
    int*,
    int*,
    double*
);

extern "C" {
    void matmul_cpu_(
        float* A,
        float* B,
        float* C,
        int* N,
        int* M,
        int* P,
        double* elapsed_ms
    );
    void matmul_gpu_(
        float* A,
        float* B,
        float* C,
        int* N,
        int* M,
        int* P,
        double* elapsed_ms
    );
}

template<matmul_func T>
std::pair<double, py::array_t<float>> matmul_wrapper(
    py::array_t<float> A,
    py::array_t<float> B
)
{
    auto bufA = A.request();
    auto bufB = B.request();

    int N = bufA.shape[0];
    int M = bufA.shape[1];
    int P = bufB.shape[1];

    py::array_t<float, py::array::f_style> C({N, P});

    auto bufC = C.request();

    double elapsed_ms = 0.0;

    T(
        static_cast<float*>(bufA.ptr),
        static_cast<float*>(bufB.ptr),
        static_cast<float*>(bufC.ptr),
        &N,
        &M,
        &P,
        &elapsed_ms
    );

    return {elapsed_ms, C};
}
#endif


PYBIND11_MODULE(hpc_benchmark_core, m) {
    m.doc() = "accelerated matrix operations";

    // Bind functions
    m.def("eve_dot", &eve_dot, "Original EVE matrix dot product",
        py::arg("A"), py::arg("B")
    );

    m.def("eve_dot2", &eve_dot2, "Optimized streaming EVE matrix dot product",
        py::arg("A"), py::arg("B")
    );

    m.def("cuda_dot", &cuda_dot, "CUDA matrix dot product",
        py::arg("A"), py::arg("B"), py::arg("blockx"), py::arg("blocky")
    );
    m.def("cuda_dot2", &cuda_dot2, "Optimized CUDA matrix dot product",
        py::arg("A"), py::arg("B"), py::arg("blockx"), py::arg("blocky")
    );

    m.def("kokkos_dot", &kokkos_dot, "Kokkos matrix multiplication execution layer",
        py::arg("A"), py::arg("B"), py::arg("blockx"), py::arg("blocky")
    );
    m.def("finalize_kokkos", &finalize_kokkos_once,
          "Finalise Kokkos avant la sortie du processus");

    m.def("eigen_dot", &eigen_dot, "Eigen CPU Matrix Multiplication",
        py::arg("A"), py::arg("B")
    );

    m.def("armadillo_dot", &eigen_dot, "ARMADILLO CPU Matrix Multiplication",
        py::arg("A"), py::arg("B")
    );
    m.def("armadillo_dot_gpu", &armadillo_dot_gpu, "ARMADILLO GPU Matrix Multiplication",
        py::arg("A"), py::arg("B")
    );

    m.def("kiwaku_dot", &kiwaku_dot, "Kiwaku Layout Managed CPU Matrix Multiplication",
        py::arg("A"), py::arg("B")
    );
    m.def("kiwaku_dot2", &kiwaku_dot2, "Optimized Kiwaku Layout Managed CPU Matrix Multiplication",
        py::arg("A"), py::arg("B")
    );

#ifdef HAS_FORTRAN
    m.def("matmul_cpu", &matmul_wrapper<matmul_cpu_>, "Fortran CPU matrix multiplication",
        py::arg("A"), py::arg("B")
    );

    m.def("matmul_gpu", &matmul_wrapper<matmul_gpu_>, "Fortran GPU matrix multiplication",
        py::arg("A"), py::arg("B")
    );
#endif
}
