#pragma once
#include <vector>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::pair<double, py::array_t<float>> cuda_dot(
    py::array_t<float> A,
    py::array_t<float> B
);