// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

#pragma once
#include "matrix_dot.hu"
#include "cuda_wrapper.h"


// Utilise le noyau standard
std::pair<double, py::array_t<float>> cuda_dot(
    py::array_t<float> A,
    py::array_t<float> B,
    int blockx, int blocky
)
{
    return cuda_dot_T<launch_matmul_kernel>(A, B, blockx, blocky);
}

// Utilise le noyau optimisé avec shared memory / tiling
std::pair<double, py::array_t<float>> cuda_dot2(
    py::array_t<float> A,
    py::array_t<float> B,
    int blockx, int blocky
)
{
    return cuda_dot_T<launch_matmul_kernel_optimized>(A, B, blockx, blocky);
}