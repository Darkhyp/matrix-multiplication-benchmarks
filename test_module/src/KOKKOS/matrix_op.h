#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

std::pair<double, py::array_t<float>> kokkos_dot(
    py::array_t<float> A,
    py::array_t<float> B
);