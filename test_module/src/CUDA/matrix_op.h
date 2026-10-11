// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

#pragma once
#include <vector>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::pair<double, py::array_t<float>> cuda_dot(
    py::array_t<float> A,
    py::array_t<float> B,
    int blockx,
    int blocky
);