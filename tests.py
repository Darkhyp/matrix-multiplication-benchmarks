# Copyright (c) 2026
# Author: Alexandre Korovin
# e-mail: a.v.korovin73@gmail.com
# SPDX-License-Identifier: MIT
import os
import platform
import sys

import numpy as np
import cupy as cp
import time
from pathlib import Path

from test_functions.benchmark import benchmark
from test_functions.numba_test import numba_matrix_dot
from test_functions.pyCUDA_test import pyCUDA_matrix_dot, ctx
from test_functions.tools import print_system_info

if platform.system() == "Linux":
    print(f"{Path(__file__).name}: Running on Linux")
else:
    print(f"{Path(__file__).name}: Running on Windows")

current_dir = Path(__file__).parent
test_module_dir = current_dir / "test_module"
if test_module_dir.is_dir() and str(test_module_dir) not in sys.path:
    sys.path.insert(0, str(test_module_dir))
import hpc_benchmark_core as TEST

# -------------------------------------------------------------------------
# Warmup Calls (Important: Numba/EVE/GPU compiles on the very first execution)
# -------------------------------------------------------------------------
WARMUP = True
# WARMUP = False
N_repetitions = 50
# blockx, blocky = 16, 8
blockx, blocky = 16, 16
# blockx, blocky = 32, 16
# blockx, blocky = 16, 32
# blockx, blocky = 8, 64
# blockx, blocky = 64, 8

# Define dimensions
N, M, P = 512, 512 + 1, 512 + 2


@benchmark("numpy (CPU) Loop", n=N_repetitions, warmup=WARMUP)
def numpy_test(A, B):
    start_time = time.perf_counter()
    C = np.dot(A, B)
    execution_time = (time.perf_counter() - start_time) * 1000  # time in ms
    return execution_time, C

@benchmark("CuPy (GPU) Loop", n=N_repetitions, warmup=WARMUP)
def cupy_test(A, B):
    start_time = time.perf_counter()
    C = cp.dot(A, B)
    cp.cuda.Stream.null.synchronize()
    execution_time = (time.perf_counter() - start_time) * 1000  # time in ms
    return execution_time, C.get()

@benchmark("Numba JIT Loop", n=N_repetitions, warmup=WARMUP)
def numba_test(A, B):
    start_time = time.perf_counter()
    C = numba_matrix_dot(A, B)
    execution_time = (time.perf_counter() - start_time) * 1000  # time in ms
    return execution_time, C

@benchmark("CUDA cu (GPU) Loop", n=N_repetitions, warmup=WARMUP)
def cuda_test(A, B):
    return TEST.cuda_dot(A, B, blockx, blocky)
@benchmark("CUDA cu (GPU) Loop2", n=N_repetitions, warmup=WARMUP)
def cuda_test2(A, B):
    return TEST.cuda_dot2(A, B, blockx, blocky)

@benchmark("pyCUDA (GPU) Loop", n=N_repetitions, warmup=WARMUP)
def pycuda_test(A, B):
    return pyCUDA_matrix_dot(A, B)

@benchmark("KOKKOS C++ Engine", n=N_repetitions, warmup=WARMUP)
def kokkos_test(A, B):
    return TEST.kokkos_dot(A, B, blockx, blocky)

@benchmark("EIGEN C++ Engine", n=N_repetitions, warmup=WARMUP)
def eigen_test(A, B):
    return TEST.eigen_dot(A, B)

@benchmark("ARMADILLO CPU C++ Engine", n=N_repetitions, warmup=WARMUP)
def armadillo_test(A, B):
    return TEST.armadillo_dot(A, B)

@benchmark("cuBLAS GPU C++ Engine", n=N_repetitions, warmup=WARMUP)
def cublas_test(A, B):
    return TEST.cublas_dot_gpu(A, B)

@benchmark("EVE C++ Engine", n=N_repetitions, warmup=WARMUP)
def eve_test(A, B):
    return TEST.eve_dot(A, B)
@benchmark("EVE C++ Engine2", n=N_repetitions, warmup=WARMUP)
def eve_test2(A, B):
    return TEST.eve_dot2(A, B)

@benchmark("KIWAKU C++ Engine", n=N_repetitions, warmup=WARMUP)
def kiwaku_test(A, B):
    return TEST.kiwaku_dot(A, B)
@benchmark("KIWAKU C++ Engine2", n=N_repetitions, warmup=WARMUP)
def kiwaku_test2(A, B):
    return TEST.kiwaku_dot2(A, B)

if platform.system() == "Linux":
    @benchmark("Fortran CPU Engine", n=N_repetitions, warmup=WARMUP)
    def fortran_cpu_test(A, B):
        return TEST.matmul_cpu(A, B)

    @benchmark("Fortran GPU Engine", n=N_repetitions, warmup=WARMUP)
    def fortran_gpu_test(A, B):
        return TEST.matmul_gpu(A, B)

def readme_print(results, ref_results):
    numpy_execution_time, numpy_median_time, dnumpy_execution_time, C_numpy = ref_results["mean"], ref_results[
        "median"], ref_results["stdev"], ref_results["result"]

    print("| Engine / Backend               | Execution Time (ms) | median (ms) | Speedup vs. NumPy | Status Verification          |")
    print("|:-------------------------------|:-------------------:|:-----------:|:-----------------:|:-----------------------------|")
    print(f"| **NumPy (np.dot)**             | {numpy_execution_time:10.4f} ±{dnumpy_execution_time:7.4f} | {numpy_median_time:11.4f} | *1.00x*           | *Reference Standard*         |")

    for result in results:
        t = result["mean"]
        tm = result["median"]
        dt = result["stdev"]
        print(f'| {"**" + result["comment"]+"**":30s} | {t:10.4f} ±{dt:7.4f} | {tm:11.4f} |{numpy_execution_time/t:8.2f}({numpy_median_time/tm:7.2f})x | ', end="")
        C_test = result["result"]
        matrices_match = np.allclose(C_test, C_numpy, rtol=1e-05, atol=1e-05)
        if matrices_match:
            # Show peak difference
            max_diff = np.max(np.abs(C_test - C_numpy))
            print(f"✅ Passed (Max Dev: {max_diff:.2e}) |")
        else:
            # Diagnostics
            mismatches = np.where(~np.isclose(C_test, C_numpy, rtol=1e-05, atol=1e-05))
            print(f"❌ VERIFICATION FAILED (Max Dev: {len(mismatches[0])}) |")


if __name__ == '__main__':
    print_system_info()
    print()

    # Create two random matrices
    A = np.random.rand(N, M).astype(np.float32)
    B = np.random.rand(M, P).astype(np.float32)

    A_f = np.asfortranarray(A, dtype=np.float32)
    B_f = np.asfortranarray(B, dtype=np.float32)

    A_gpu = cp.array(A)
    B_gpu = cp.array(B)

    # gathering results
    results = []

    # Cross-check accuracy with NumPy
    print("Running numpy Matrix Multiplication...")
    ref_results = numpy_test(A, B)

    print("Running CuPy (GPU Acceleration) Matrix Multiplication...")
    results.append(cupy_test(A_gpu, B_gpu))

    try:
        # Call the C++ extension with KOKKOS
        print("Running KOKKOS Matrix Multiplication...")
        results.append(kokkos_test(A, B))
    finally:
        TEST.finalize_kokkos()

    # Call the C++ extension with CUDA
    # print("Running CUDA (GPU Acceleration) Matrix Multiplication...")
    # results.append(cuda_test(A, B))
    print("Running CUDA (GPU Acceleration) Matrix Multiplication2...")
    results.append(cuda_test2(A, B))

    # Call the python with PyCUDA
    print("Running PyCUDA Matrix Multiplication...")
    ctx.push()
    results.append(pycuda_test(A, B))
    ctx.pop()
    ctx.detach()

    # Call the C++ extension with EIGEN
    print("Running EIGEN Matrix Multiplication...")
    results.append(eigen_test(A, B))

    # Call the C++ extension with ARMADILLO
    print("Running ARMADILLO CPU Matrix Multiplication...")
    results.append(armadillo_test(A, B))

    print("Running cuBLAS GPU Matrix Multiplication...")
    results.append(cublas_test(A, B))

    # Call the python with NUMBA
    print("Running numba Matrix Multiplication...")
    results.append(numba_test(A, B))

    # Call the C++ extension with EVE
    # print("Running EVE Matrix Multiplication...")
    # results.append(eve_test(A, B))
    print("Running EVE Matrix Multiplication2...")
    results.append(eve_test2(A, B))

    # Call the C++ extension with KIWAKU
    # print("Running KIWAKU Matrix Multiplication...")
    # results.append(kiwaku_test(A, B))
    print("Running KIWAKU Matrix Multiplication2...")
    results.append(kiwaku_test2(A, B))

    # Call the fortran functions
    if platform.system() == "Linux":
        print("Running Fortran CPU Matrix Multiplication...")
        results.append(fortran_cpu_test(A_f, B_f))

        print("Running Fortran GPU Matrix Multiplication...")
        results.append(fortran_gpu_test(A_f, B_f))


    # *******************************
    #     PERFORMANCE COMPARISON
    # *******************************
    print()
    print("="*121)
    print(f" MATRIX MULTIPLICATION ({N},{M}) on ({M},{P}) PERFORMANCE COMPARISON     ")
    print("="*121)
    readme_print(sorted(results, key=lambda x: x["mean"]), ref_results)
    print("="*121)
