#!/bin/bash

# Copyright (c) 2026
# Author: Alexandre Korovin
# e-mail: a.v.korovin73@gmail.com
# SPDX-License-Identifier: MIT

set -e  # Exit instantly if any command fails

echo "=== Cleaning Old Build States ==="
# Clear everything in dist_linux EXCEPT the build/ folder itself
# (build/ is a volume mount point — can't rm the directory, only its contents)
find test_module/KOKKOS_SRC/dist_linux -mindepth 1 -maxdepth 1 ! -name build -exec rm -rf {} +
# Clear the contents of the mounted build/ volume separately
find test_module/KOKKOS_SRC/dist_linux/build -mindepth 1 -delete

echo "=== Compiling Kokkos Dependency Natively ==="
cd test_module/KOKKOS_SRC
mkdir -p dist_linux/build && cd dist_linux/build
cmake ../.. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
         -DCMAKE_CUDA_HOST_COMPILER=g++-12 \
         -DCMAKE_CXX_COMPILER=g++-12 \
         -DCMAKE_CXX_STANDARD=20 \
         -DCMAKE_BUILD_TYPE=Release \
         -DKokkos_ENABLE_CUDA=ON \
         -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON \
         -DKokkos_ARCH_ADA89=ON \
         -DCMAKE_INSTALL_PREFIX=".."
cmake --build . --config Release --target install
cd ../../../..

echo "=== Building Python C++/Fortran Extensions ==="
cd test_module_linux
rm -rf build/ *.egg-info/
python3 -m pip install -e .
cd ..

echo "=== Launching Benchmark Suite ==="
export PYTHONPATH="${PYTHONPATH}:/workspace/test_module_linux"
python3 tests.py
