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
#include "CUBLAS/matrix_op.h"
#include "FORTRAN/fortran.h"

namespace py = pybind11;


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

    m.def("cublas_dot_gpu", &cublas_dot_gpu, "cuBLAS GPU Matrix Multiplication",
        py::arg("A"), py::arg("B")
    );

    m.def("kiwaku_dot", &kiwaku_dot, "Kiwaku Layout Managed CPU Matrix Multiplication",
        py::arg("A"), py::arg("B")
    );
    m.def("kiwaku_dot2", &kiwaku_dot2, "Optimized Kiwaku Layout Managed CPU Matrix Multiplication",
        py::arg("A"), py::arg("B")
    );

    m.def("matmul_cpu", &matmul_wrapper<matmul_cpu_>, "Fortran CPU matrix multiplication",
        py::arg("A"), py::arg("B")
    );
    m.def("matmul_cpu_sgemm", &matmul_wrapper<matmul_cpu_sgemm_>, "Fortran CPU matrix multiplication",
        py::arg("A"), py::arg("B")
    );

#ifdef HAS_GPU_FORTRAN
    m.def("matmul_gpu", &matmul_wrapper<matmul_gpu_>, "Fortran GPU matrix multiplication",
        py::arg("A"), py::arg("B")
    );
#endif
}
