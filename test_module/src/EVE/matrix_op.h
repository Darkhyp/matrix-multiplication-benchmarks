#pragma once
#include <vector>
#include <pybind11/numpy.h>

namespace py = pybind11;

std::pair<double, py::array_t<float>> eve_dot(
    py::array_t<float> A,
    py::array_t<float> B
);

// modified dot function
std::pair<double, py::array_t<float>> eve_dot2(
    py::array_t<float> A,
    py::array_t<float> B
);
