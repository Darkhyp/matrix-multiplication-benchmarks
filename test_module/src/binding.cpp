// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

 #include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "EVE/matrix_op.h"
#include "CUDA/matrix_op.h"
#include "KOKKOS/matrix_op.h"
#include "KIWAKU/matrix_op.h"

namespace py = pybind11;

PYBIND11_MODULE(test_module, m) {
    m.doc() = "accelerated matrix operations";

    // Bind functions
    m.def("eve_dot", &eve_dot, "Original EVE matrix dot product",
        py::arg("A"), py::arg("B")
    );

    m.def("eve_dot2", &eve_dot2, "Optimized streaming EVE matrix dot product",
        py::arg("A"), py::arg("B")
    );

    m.def("cuda_dot", &cuda_dot, "Optimized streaming EVE matrix dot product",
        py::arg("A"), py::arg("B")
    );

    m.def("kokkos_dot", &kokkos_dot, "Kokkos matrix multiplication execution layer",
        py::arg("A"), py::arg("B")
    );

    m.def("kiwaku_dot", &kiwaku_dot, "Kiwaku Layout Managed CPU Matrix Multiplication",
        py::arg("A"), py::arg("B")
    );
}
