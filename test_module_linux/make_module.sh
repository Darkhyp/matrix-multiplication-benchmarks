#!/bin/bash

# Copyright (c) 2026
# Author: Alexandre Korovin
# e-mail: a.v.korovin73@gmail.com
# SPDX-License-Identifier: MIT

set -e  # Exit instantly if any command fails
source "print_box.sh"

WORK=/workspace

KOKKOS_SRC=${WORK}/test_module/KOKKOS_SRC
KOKKOS_BUILD=/opt/kokkos/build
KOKKOS_INSTALL=/opt/kokkos

# Vérifier les sources
if [[ ! -f "${KOKKOS_SRC}/CMakeLists.txt" ]]; then
    echo "ERROR: Kokkos sources not found: ${KOKKOS_SRC}" >&2
    exit 1
fi


if [[ ! -f "${KOKKOS_INSTALL}/lib/libkokkoscore.a" &&
      ! -f "${KOKKOS_INSTALL}/lib64/libkokkoscore.a" ]]; then

    print_box "Preparing Kokkos Build"

    # Créer le répertoire avant cd
    mkdir -p "${KOKKOS_BUILD}" || exit 1
    mkdir -p "${KOKKOS_INSTALL}" || exit 1

    print_box "Compiling Kokkos Dependency Natively"
    cd "$KOKKOS_BUILD" || exit 1

    cmake "$KOKKOS_SRC" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_CUDA_HOST_COMPILER=g++-12 \
        -DCMAKE_CXX_COMPILER=g++-12 \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_BUILD_TYPE=Release \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON \
        -DKokkos_ARCH_ADA89=ON \
        -DCMAKE_INSTALL_PREFIX="$KOKKOS_INSTALL" || exit 1

    cmake --build . --config Release --target install || exit 1

    cd ${WORK}

else
    print_box "Kokkos already compiled - skipping build"
fi

print_box "Building/Rebuilding Python C++/Fortran Extensions"
cd ${WORK}/test_module_linux
rm -rf build/ *.egg-info/ *.so
#pip install -e .
pip install -e . --no-build-isolation -v

# doit lister hpc_benchmark_core*.so ET libmatmul_acc.so
ls -l *.so
ldd hpc_benchmark_core*.so | grep -Ei "matmul_acc|acc|nvf|cuda"

#export PYTHONPATH="${PYTHONPATH}:${WORK}/test_module_linux"
cd ${WORK}

print_box "Launching Benchmark Suite"
python3 tests.py
#NV_ACC_NOTIFY=3 python3 tests.py   # chaque lancement de kernel et chaque transfert
#NV_ACC_TIME=1  python3 tests.py    # bilan de temps par région à la sortie
