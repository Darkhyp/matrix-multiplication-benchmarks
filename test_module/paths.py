# Copyright (c) 2026
# Author: Alexandre Korovin
# e-mail: a.v.korovin73@gmail.com
# SPDX-License-Identifier: MIT

import os
import sys
import sysconfig
from pathlib import Path

def path2str(paths: list[Path]):
    return [str(Path(p).resolve()) for p in paths]

# Définition des racines selon l'OS
if sys.platform == "win32":
    MODULE_ROOT = Path(__file__).resolve().parent
else:
    import pybind11

    ROOT = Path("/workspace")
    MODULE_ROOT = ROOT / "test_module"
SRC = MODULE_ROOT / "src"

# Configuration spécifique par OS
if sys.platform == "win32":
    MSVC_BIN = r"C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Tools\MSVC\14.50.35717\bin\HostX86\x64"
    MSYS_LIB = Path(r"C:\msys64\ucrt64\lib")

    LIBGFORTRAN = str(MSYS_LIB / "libgfortran.a")
    LIBQUADMATH = str(MSYS_LIB / "libquadmath.a")

    OPENBLAS_ROOT = MODULE_ROOT / "vcpkg" / "installed" / "x64-windows-static-release"
    OPENBLAS_INCLUDE = OPENBLAS_ROOT / "include"
    OPENBLAS_LIB = OPENBLAS_ROOT / "lib"

    CUDA_PATH = Path(os.environ.get(
        "CUDA_PATH",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    ))
    CUDA_LIB = CUDA_PATH / "lib" / "x64"
    NVCC = CUDA_PATH / "bin" / "nvcc.exe"

    KOKKOS_ROOT = MODULE_ROOT / "KOKKOS_SRC" / "dist"

else:
    OPENBLAS_INCLUDE = OPENBLAS_LIB = LAPACK_INCLUDE = LAPACK_LIB = Path("")

    CUDA_PATH = Path("/usr/local/cuda")
    CUDA_LIB = CUDA_PATH / "lib64"
    NVCC = CUDA_PATH / "bin" / "nvcc"

    KOKKOS_ROOT = Path("/opt/kokkos")

    NVHPC_ROOT = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/24.5")
    NVHPC_COMPILER = NVHPC_ROOT / "compilers" / "bin"
    NVHPC_LIB = NVHPC_ROOT / "compilers" / "lib"
    NVFORTRAN = NVHPC_COMPILER / "nvfortran"

    GFORTRAN = "/usr/bin/gfortran-12"

    GPU_LIB_NAME = "matmul_acc"  # -> libmatmul_acc.so
    GPU_LIB_FILE = f"lib{GPU_LIB_NAME}.so"


# include pathes
EVE_INCLUDE = MODULE_ROOT / "EVE_SRC" / "include"
EIGEN_INCLUDE = MODULE_ROOT / "EIGEN_SRC"
ARMADILLO_INCLUDE = MODULE_ROOT / "ARMADILLO_SRC" / "include"
KIWAKU_INCLUDE = MODULE_ROOT / "KIWAKU_SRC" / "include"
RABERU_INCLUDE = MODULE_ROOT / "RABERU_SRC" / "include"
KUMI_INCLUDE = MODULE_ROOT / "KUMI_SRC" / "include"
KOKKOS_INCLUDE = KOKKOS_ROOT / "include"
KOKKOS_LIB = KOKKOS_ROOT / "lib"
CUDA_INCLUDE = CUDA_PATH / "include"

# ------------------------------------------------------------
# Fortran sources
# ------------------------------------------------------------

FORTRAN_CPU = [
    SRC / "FORTRAN" / "matrix_dot.f90",
]

FORTRAN_GPU = [
    SRC / "FORTRAN" / "matrix_dot2.f90",
]

# ------------------------------------------------------------
# C++ sources
# ------------------------------------------------------------

CPP_SOURCES = path2str([
    SRC / "binding.cpp",
    SRC / "KIWAKU" / "matrix_dot.cpp",
    SRC / "KIWAKU" / "matrix_dot2.cpp",
    SRC / "EVE" / "matrix_dot.cpp",
    SRC / "EVE" / "matrix_dot2.cpp",
    SRC / "EIGEN" / "matrix_dot.cpp",
    SRC / "ARMADILLO" / "matrix_dot.cpp",
    SRC / "ARMADILLO" / "matrix_dot2.cpp",
    SRC / "CUDA" / "matrix_op.cpp",
])

# --- CUDA (nvcc) ---
CUDA_SOURCES = [
    SRC / "CUDA" / "matrix_dot_kernel.cu",
    SRC / "CUDA" / "matrix_dot_kernel2.cu",
    SRC / "KOKKOS" / "matrix_dot.cu",
]

# ------------------------------------------------------------
# Include directories
# ------------------------------------------------------------

INCLUDE_DIRS = path2str([
    SRC,
    OPENBLAS_INCLUDE,
    KOKKOS_INCLUDE,
    EVE_INCLUDE,
    EIGEN_INCLUDE,
    ARMADILLO_INCLUDE,
    KIWAKU_INCLUDE,
    RABERU_INCLUDE,
    KUMI_INCLUDE,
])
if sys.platform == "Linux":
    INCLUDE_DIRS += [
        pybind11.get_include(),
        pybind11.get_include(user=True),
        sysconfig.get_path("include"),
    ]

# ------------------------------------------------------------
# Lib directories
# ------------------------------------------------------------

LIBRARY_DIRS = path2str([
    KOKKOS_LIB,
    CUDA_LIB,
    OPENBLAS_LIB,
])
