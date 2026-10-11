# Copyright (c) 2026
# Author: Alexandre Korovin
# SPDX-License-Identifier: MIT

import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pybind11
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

ROOT = Path("/workspace")
SRC = ROOT / "test_module" / "src"
PACKAGE_DIR = ROOT / "test_module_linux"

KOKKOS_DIR = Path("/opt/kokkos")
KOKKOS_INCLUDE = Path("/opt/kokkos/include")
KOKKOS_LIB = Path("/opt/kokkos/lib")

CUDA_ROOT = Path("/usr/local/cuda")
CUDA_INCLUDE = CUDA_ROOT / "include"
CUDA_LIB_DIR = CUDA_ROOT / "lib64"
NVCC = CUDA_ROOT / "bin" / "nvcc"

NVHPC_ROOT = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/24.5")
NVHPC_COMPILER_DIR = NVHPC_ROOT / "compilers" / "bin"
NVHPC_LIB = NVHPC_ROOT / "compilers" / "lib"
NVFORTRAN = NVHPC_COMPILER_DIR / "nvfortran"

GFORTRAN = "/usr/bin/gfortran-12"

GPU_LIB_NAME = "matmul_acc"                 # -> libmatmul_acc.so
GPU_LIB_FILE = f"lib{GPU_LIB_NAME}.so"

os.environ["CC"] = "/usr/bin/gcc"
os.environ["CXX"] = "/usr/bin/g++"


# ------------------------------------------------------------
# Include directories
# ------------------------------------------------------------

INCLUDE_DIRS = [
    str(SRC),
    str(CUDA_INCLUDE),
    str(KOKKOS_INCLUDE),
    str(ROOT / "test_module" / "EVE_SRC" / "include"),
    str(ROOT / "test_module" / "KIWAKU_SRC" / "include"),
    str(ROOT / "test_module" / "RABERU_SRC" / "include"),
    str(ROOT / "test_module" / "KUMI_SRC" / "include"),
    pybind11.get_include(),
    pybind11.get_include(user=True),
    sysconfig.get_path("include"),
]

LIBRARY_DIRS = [
    str(KOKKOS_LIB),
    str(CUDA_LIB_DIR),
    str(NVHPC_LIB),
]


# ------------------------------------------------------------
# Fortran compilation
# ------------------------------------------------------------

FORTRAN_CPU = [
    SRC / "FORTRAN" / "matrix_dot.f90",
]

FORTRAN_GPU = [
    SRC / "FORTRAN" / "matrix_dot2.f90",
]


def compile_fortran_cpu(sources, output_dir):
    """Compile les sources Fortran CPU (gfortran) en objets PIC."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flags = ["-O3", "-fPIC", "-std=f2018"]
    objects = []

    for source in sources:
        source = Path(source)

        if not source.is_file():
            raise FileNotFoundError(f"Fortran source does not exist: {source}")

        obj = output_dir / f"{source.stem}.o"
        command = [GFORTRAN, "-c", str(source), "-o", str(obj), *flags]

        print("\nCompiling Fortran (CPU):")
        print(" ".join(command), flush=True)

        subprocess.run(command, check=True)
        objects.append(str(obj))

    return objects


def build_fortran_gpu_shared(sources, output_dir):
    """
    Compile + lie les sources OpenACC avec nvfortran en bibliotheque
    partagee. C'est nvfortran qui fait l'etape de device-link (nvlink)
    et qui enregistre le binaire GPU aupres du runtime OpenACC.
    """

    if not NVFORTRAN.is_file():
        raise RuntimeError(f"nvfortran not found: {NVFORTRAN}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for source in sources:
        if not Path(source).is_file():
            raise FileNotFoundError(f"Fortran source does not exist: {source}")

    lib = output_dir / GPU_LIB_FILE

    command = [
        str(NVFORTRAN),
        "-O3",
        "-fPIC",
        "-shared",
        "-acc=gpu",
        "-gpu=cuda12.4,cc89",
        *map(str, sources),
        "-o", str(lib),
        f"-Wl,-rpath,{NVHPC_LIB}",
        f"-Wl,-rpath,{CUDA_LIB_DIR}",
    ]

    print("\nBuilding Fortran GPU shared library:")
    print(" ".join(command), flush=True)

    subprocess.run(command, check=True)
    return output_dir, lib


# ------------------------------------------------------------
# C++ / CUDA extension
# ------------------------------------------------------------

class BuildCUDALinux(build_ext):

    def build_extensions(self):
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        # --- Fortran CPU (objets) ---
        fortran_objects = compile_fortran_cpu(
            FORTRAN_CPU,
            build_temp / "fortran_cpu",
        )

        # --- Fortran GPU (lib partagee device-linkee) ---
        gpu_lib_dir, gpu_lib = build_fortran_gpu_shared(
            FORTRAN_GPU,
            build_temp / "fortran_gpu",
        )
        self._gpu_lib = gpu_lib

        # --- CUDA (nvcc) ---
        cuda_sources = [
            SRC / "CUDA" / "matrix_dot_kernel.cu",
            SRC / "KOKKOS" / "matrix_dot.cu",
        ]

        cuda_objects = []

        for source in cuda_sources:
            if not source.is_file():
                raise FileNotFoundError(f"CUDA source does not exist: {source}")

            obj = build_temp / f"{source.stem}.o"

            command = [
                str(NVCC),
                "-c", str(source),
                "-o", str(obj),
                "-std=c++20",
                "--extended-lambda",
                "-O3",
                "-Xcompiler", "-fPIC",
                "-I", str(CUDA_INCLUDE),
                "-I", str(KOKKOS_DIR / "include"),
                "-I", pybind11.get_include(),
                "-I", sysconfig.get_path("include"),
                "-gencode=arch=compute_89,code=sm_89",
            ]

            print("\nCompiling CUDA:")
            print(" ".join(command), flush=True)

            subprocess.run(command, check=True)
            cuda_objects.append(str(obj))

        # --- Branchement sur l'extension ---
        for ext in self.extensions:
            ext.extra_objects.extend(fortran_objects)
            ext.extra_objects.extend(cuda_objects)

            ext.library_dirs.append(str(gpu_lib_dir))
            if GPU_LIB_NAME not in ext.libraries:
                ext.libraries.append(GPU_LIB_NAME)

            # $ORIGIN : la lib est copiee a cote du .so Python.
            # gpu_lib_dir : repli pour les builds en place / debug.
            for rp in ("$ORIGIN", str(PACKAGE_DIR)):
                if rp not in ext.runtime_library_dirs:
                    ext.runtime_library_dirs.append(rp)

        super().build_extensions()

        # Copie libmatmul_acc.so à côté de l'extension construite
        # ET dans PACKAGE_DIR (stable, survit au build temporaire de pip -e).
        for ext in self.extensions:
            dest_dirs = {
                Path(self.get_ext_fullpath(ext.name)).resolve().parent,
                PACKAGE_DIR.resolve(),
            }
            for dest_dir in dest_dirs:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / GPU_LIB_FILE
                print(f"\nCopying {gpu_lib} -> {dest}", flush=True)
                shutil.copy2(gpu_lib, dest)


# ------------------------------------------------------------
# Extension sources
# ------------------------------------------------------------

cpp_sources = [
    str(SRC / "binding_linux.cpp"),
    str(SRC / "KIWAKU" / "matrix_dot.cpp"),
    str(SRC / "EVE" / "matrix_dot.cpp"),
    str(SRC / "EVE" / "matrix_dot2.cpp"),
    str(SRC / "CUDA" / "matrix_op.cpp"),
]

for source in cpp_sources:
    if not Path(source).is_file():
        raise FileNotFoundError(f"C++ source does not exist: {source}")


extension = Pybind11Extension(
    "hpc_benchmark_core",
    cpp_sources,
    include_dirs=INCLUDE_DIRS,
    library_dirs=LIBRARY_DIRS,
    # Les libs NVHPC (acchost, nvf, nvomp, nvc, nvcpumath, nvhpcatm...)
    # sont desormais liees dans libmatmul_acc.so : inutile de les
    # relier ici (et cela evite les doublons de runtime).
    libraries=[
        "kokkoscore",
        "cudart",
        "cuda",
        "openblas",
        "gfortran",
        "quadmath",
        "dl",
        "pthread",
        "m",
    ],
    cxx_std=20,
    extra_compile_args=[
        "-O3",
        "-fPIC",
        "-mavx2",
    ],
    extra_link_args=[
        "-Wl,--no-as-needed",
        f"-Wl,-rpath,{NVHPC_LIB}",
    ],
    runtime_library_dirs=[
        str(NVHPC_LIB),
        str(CUDA_LIB_DIR),
    ],
)


# ------------------------------------------------------------
# Package
# ------------------------------------------------------------

setup(
    name="hpc_test_extensions",
    version="1.0.0",
    ext_modules=[extension],
    cmdclass={
        "build_ext": BuildCUDALinux,
    },
    install_requires=[
        "numpy",
        "cupy-cuda12x",
        "numba",
        "pycuda",
        "psutil",
    ],
)