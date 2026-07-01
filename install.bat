pip install pybind11 numpy numba cupy-cuda12x pycuda psutil

:: ***********************
::      INSTALL EVE
:: ***********************
@REM vcpkg install eve
git clone https://github.com/jfalcou/eve.git test_module/EVE_SRC

:: ***********************
::     INSTALL KIWAKU
:: ***********************
git clone https://github.com/jfalcou/kiwaku.git test_module/KIWAKU_SRC
git clone https://github.com/jfalcou/raberu.git test_module/RABERU_SRC
git clone https://github.com/jfalcou/kumi.git test_module/KUMI_SRC

:: ***********************
::     INSTALL KOKKOS
:: ***********************
git clone --recursive https://github.com/kokkos/kokkos.git test_module/KOKKOS_SRC
cd test_module/KOKKOS_SRC
rmdir /s /q build
mkdir build
cd build

cmake .. -DCMAKE_CXX_STANDARD=20 -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON -DKokkos_ARCH_ADA89=ON -DCMAKE_INSTALL_PREFIX="F:/usr/PycharmProjects/VEC_tests/test_module/KOKKOS_SRC/dist"
cmake --build . --config Release --target install

:: ***********************
::    INSTALL nvfortran
:: ***********************
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_nvhpc.ps1"
pause