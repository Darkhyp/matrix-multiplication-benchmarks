:: Copyright (c) 2026
:: Author: Alexandre Korovin
:: e-mail: a.v.korovin73@gmail.com
:: SPDX-License-Identifier: MIT

call print_box.bat "INSTALL Python libs"
pip install pybind11 numpy numba cupy-cuda12x pycuda psutil

call print_box.bat "INSTALL EVE"
@REM vcpkg install eve
git clone https://github.com/jfalcou/eve.git test_module/EVE_SRC

call print_box.bat "INSTALL KIWAKU"
git clone https://github.com/jfalcou/kiwaku.git test_module/KIWAKU_SRC
git clone https://github.com/jfalcou/raberu.git test_module/RABERU_SRC
git clone https://github.com/jfalcou/kumi.git test_module/KUMI_SRC

call print_box.bat "INSTALL KOKKOS"
git clone --recursive https://github.com/kokkos/kokkos.git test_module/KOKKOS_SRC
cd test_module/KOKKOS_SRC
rmdir /s /q build
mkdir build

call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat"
where cl

cmake -S . -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CXX_COMPILER=cl ^
  -DCMAKE_CXX_STANDARD=20 ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" ^
  -DKokkos_ENABLE_CUDA=ON ^
  -DKokkos_ENABLE_CUDA_LAMBDA=ON ^
  -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON ^
  -DKokkos_ENABLE_SERIAL=ON ^
  -DKokkos_ARCH_ADA89=ON ^
  -DCMAKE_INSTALL_PREFIX=F:/usr/PycharmProjects/VEC_tests/test_module/KOKKOS_SRC/dist

cmake --build build --target install

call print_box.bat "INSTALL nvfortran"
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_nvhpc.ps1"
pause
