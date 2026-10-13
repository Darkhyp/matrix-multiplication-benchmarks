:: Copyright (c) 2026
:: Author: Alexandre Korovin
:: e-mail: a.v.korovin73@gmail.com
:: SPDX-License-Identifier: MIT

@echo off
SETLOCAL Enabledelayedexpansion
chcp 65001 >nul

set "ROOT_DIR=%~dp0"
set "MODULE_DIR=%ROOT_DIR%\test_module"


call print_box.bat "INSTALL Python libs"
pip install pybind11 numpy numba cupy-cuda12x pycuda psutil

@REM call print_box.bat "INSTALL Openblas libs"
@REM if not exist "%MODULE_DIR%\OPENBLAS_SRC" (
@REM     echo [INFO] OpenBLAS cloning...
@REM     git clone https://github.com/OpenMathLib/OpenBLAS.git test_module/OPENBLAS_SRC
@REM
@REM @REM git clone https://github.com/Microsoft/vcpkg.git test_module/vcpkg
@REM @REM cd test_module/vcpkg
@REM @REM bootstrap-vcpkg.bat
@REM @REM vcpkg install openblas:x64-windows-static-release
@REM @REM vcpkg install lapack:x64-windows-static-release
@REM
@REM ) else (echo [INFO] OpenBLAS exist)
@REM if not exist "%MODULE_DIR%\OPENBLAS_SRC/dist/lib/openblas.lib" (
@REM     echo [INFO] OpenBLAS compiling...
@REM
@REM     cd "%MODULE_DIR%\OPENBLAS_SRC"
@REM     rmdir /s /q build
@REM     mkdir build
@REM
@REM     call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat"
@REM     where cl
@REM
@REM     cmake -S . -B build -G Ninja ^
@REM         -DCMAKE_C_COMPILER=cl ^
@REM         -DCMAKE_CXX_COMPILER=cl ^
@REM         -DCMAKE_BUILD_TYPE=Release ^
@REM         -DNOFORTRAN=ON ^
@REM         -DDYNAMIC_ARCH=ON ^
@REM         -DCMAKE_INSTALL_PREFIX="%MODULE_DIR%\OPENBLAS_SRC\dist"
@REM
@REM     cmake --build build --config Release --target install
@REM
@REM     cd %ROOT_DIR%
@REM ) else (echo [INFO] OpenBLAS lib exist)

call print_box.bat "INSTALL Eigen libs"
if not exist "%MODULE_DIR%\EIGEN_SRC" (
    echo [INFO] EIGEN cloning...
    @REM vcpkg install eigen3
    git clone https://gitlab.com/libeigen/eigen.git test_module/EIGEN_SRC
) else (echo [INFO] EIGEN exist)

call print_box.bat "INSTALL Armadillo libs"
if not exist "%MODULE_DIR%\ARMADILLO_SRC" (
    echo [INFO] ARMADILLO cloning...
    @REM vcpkg install armadillo
    git clone https://gitlab.com/conradsnicta/armadillo-code.git test_module/ARMADILLO_SRC
) else (echo [INFO] ARMADILLO exist)

call print_box.bat "INSTALL EVE"
if not exist "%MODULE_DIR%\EVE_SRC" (
    echo [INFO] EVE cloning...
    @REM vcpkg install eve
    git clone https://github.com/jfalcou/eve.git test_module/EVE_SRC
) else (echo [INFO] EVE exist)

call print_box.bat "INSTALL KIWAKU"
if not exist "%MODULE_DIR%\KIWAKU_SRC" (
    echo [INFO] KIWAKU cloning...
    git clone https://github.com/jfalcou/kiwaku.git test_module/KIWAKU_SRC
) else (echo [INFO] KIWAKU exist)
if not exist "%MODULE_DIR%\RABERU_SRC" (
    echo [INFO] RABERU cloning...
    git clone https://github.com/jfalcou/raberu.git test_module/RABERU_SRC
) else (echo [INFO] RABERU exist)
if not exist "%MODULE_DIR%\KUMI_SRC" (
    echo [INFO] KUMI cloning...
    git clone https://github.com/jfalcou/kumi.git test_module/KUMI_SRC
) else (echo [INFO] KUMI exist)

call print_box.bat "INSTALL KOKKOS"
if not exist "%MODULE_DIR%\KOKKOS_SRC" (
    echo [INFO] KOKKOS cloning...
    git clone --recursive https://github.com/kokkos/kokkos.git test_module/KOKKOS_SRC
) else (echo [INFO] KOKKOS exist)
set "FOUND_KOKKOS=0"
for %%f in ("%MODULE_DIR%\KOKKOS_SRC\dist\lib\kokkos*.lib") do (set "FOUND_KOKKOS=1")
if "%FOUND_KOKKOS%"=="0" (
    echo [INFO] KOKKOS compiling...

    cd "%MODULE_DIR%\KOKKOS_SRC"
    rmdir /s /q build
    mkdir build

    call "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat"
    where cl

    cmake -S . -B build -G Ninja ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DCMAKE_CXX_COMPILER=cl ^
        -DCMAKE_CXX_STANDARD=20 ^
        -DCMAKE_CUDA_ARCHITECTURES=89 ^
        -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -Xcompiler /MT" ^
        -DCMAKE_CXX_FLAGS_RELEASE="/MT /O2 /Ob2 /DNDEBUG" ^
        -DCMAKE_C_FLAGS_RELEASE="/MT /O2 /Ob2 /DNDEBUG" ^
        -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded" ^
        -DKokkos_ENABLE_CUDA=ON ^
        -DKokkos_ENABLE_CUDA_LAMBDA=ON ^
        -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON ^
        -DKokkos_ENABLE_SERIAL=ON ^
        -DKokkos_ARCH_ADA89=ON ^
        -DCMAKE_INSTALL_PREFIX="%MODULE_DIR%\KOKKOS_SRC\dist"

    cmake --build build --target install

    cd %ROOT_DIR%
) else (echo [INFO] KOKKOS lib exist)

@REM call print_box.bat "INSTALL nvfortran"
@REM @echo off
@REM powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_nvhpc.ps1"
@REM pause
