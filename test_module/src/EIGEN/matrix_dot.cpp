// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <chrono>

namespace py = pybind11;

std::pair<double, py::array_t<float>> eigen_dot(
    py::array_t<float> A,
    py::array_t<float> B
)
{
    const auto rows_A = A.shape(0);
    const auto cols_A = A.shape(1);
    const auto rows_B = B.shape(0);
    const auto cols_B = B.shape(1);

    if (A.ndim() != 2 || B.ndim() != 2)
        throw std::runtime_error("Expected 2D arrays");
    if (cols_A != rows_B)
        throw std::runtime_error("A and B are not compatible");

    // Création de la matrice de sortie C
    py::array_t<float> C({rows_A, cols_B});

    // Mappage des tampons NumPy vers des matrices Eigen (zéro copie)
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(
        static_cast<const float*>(A.request().ptr), rows_A, cols_A
    );
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matB(
        static_cast<const float*>(B.request().ptr), rows_B, cols_B
    );
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matC(
        static_cast<float*>(C.request().ptr), rows_A, cols_B
    );

    auto start = std::chrono::steady_clock::now();

    // Multiplication optimisée par Eigen (utilise SIMD et multi-threading automatique)
    matC.noalias() = matA * matB;

    auto end = std::chrono::steady_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return {duration_ms, C};
}
