#pragma once
#include <pybind11/numpy.h>

namespace py = pybind11;

std::pair<double, py::array_t<float>> kiwaku_dot(
    py::array_t<float> A,
    py::array_t<float> B
);