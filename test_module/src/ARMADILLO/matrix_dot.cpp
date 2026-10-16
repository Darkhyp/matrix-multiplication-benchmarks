// Copyright (c) 2026
// Author: Alexandre Korovin
// e-mail: a.v.korovin73@gmail.com
// SPDX-License-Identifier: MIT

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <armadillo>
#include <chrono>

namespace py = pybind11;

std::pair<double, py::array_t<float>> armadillo_dot(
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

    // Création de la matrice de sortie C pour NumPy
    py::array_t<float> C({rows_A, cols_B});

    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();
    py::buffer_info bufC = C.request();

    // Mappage direct des tampons NumPy vers des vues Armadillo (zéro copie)
    const float* ptr_A = static_cast<const float*>(bufA.ptr);
    const float* ptr_B = static_cast<const float*>(bufB.ptr);
    float* ptr_C = static_cast<float*>(bufC.ptr);

    // Méthode propre avec Armadillo (en supposant des matrices contiguës)
    // Note: Armadillo utilise par défaut le format column-major.
    // Pour du row-major, on utilise arma::fmat avec un constructeur de vue si besoin,
    // ou on mappe en tenant compte de la disposition mémoire.
    arma::fmat matA(const_cast<float*>(ptr_A), rows_A, cols_A, false, true); // false = don't copy, true = strict
    arma::fmat matB(const_cast<float*>(ptr_B), rows_B, cols_B, false, true);
    arma::fmat matC(ptr_C, rows_A, cols_B, false, true);

    auto start = std::chrono::steady_clock::now();

    // Multiplication (Armadillo délègue à BLAS/LAPACK si disponible, sinon utilise son propre moteur optimisé)
    matC = matA * matB;

    auto end = std::chrono::steady_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return {duration_ms, C};
}