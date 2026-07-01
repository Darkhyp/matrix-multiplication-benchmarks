import numpy as np
import cupy as cp
import test_module as TEST
import time
from numba import njit, prange

from test_module.numba_test import numba_matrix_dot
from test_module.pyCUDA_test import pyCUDA_matrix_dot
from test_module.tools import print_system_info

# -------------------------------------------------------------------------
# Warmup Calls (Important: Numba/EVE/GPU compiles on the very first execution)
# -------------------------------------------------------------------------
WARMUP = True
# WARMUP = False

if __name__ == '__main__':
    print_system_info()
    print()

    # Define dimensions
    N, M, P = 512, 512+1, 512+2

    # Create two random matrices
    A = np.random.rand(N, M).astype(np.float32)
    B = np.random.rand(M, P).astype(np.float32)

    print("Running numpy Matrix Multiplication...")
    # Cross-check accuracy with NumPy
    start_time = time.perf_counter()
    C_numpy = np.dot(A, B)
    numpy_execution_time = (time.perf_counter() - start_time) * 1000


    # gathering results
    results = []

    print("Running CuPy (GPU Acceleration) Matrix Multiplication...")
    A_gpu = cp.array(A)
    B_gpu = cp.array(B)
    if WARMUP:
        _ = cp.dot(A_gpu, B_gpu)  # Force initial kernel compilation
    cp.cuda.Stream.null.synchronize()  # Ensure warmup is complete

    start_time = time.perf_counter()
    C_gpu = cp.dot(A_gpu, B_gpu)
    cp.cuda.Stream.null.synchronize()
    cupy_execution_time = (time.perf_counter() - start_time) * 1000

    C_cupy_cpu = C_gpu.get()
    results.append({"time": cupy_execution_time, "result": C_cupy_cpu, "comment": "CuPy (GPU)  Loop"})


    print("Running CUDA (GPU Acceleration) Matrix Multiplication...")
    if WARMUP:
        _, _ = TEST.cuda_dot(A, B)
    CUDA_execution_time, result = TEST.cuda_dot(A, B)
    cupy_execution_time = (time.perf_counter() - start_time) * 1000
    results.append({"time": CUDA_execution_time, "result": result, "comment": "CUDA (GPU)  Loop"})


    print("Running PyCUDA Matrix Multiplication...")
    pyCUDA_execution_time, C_pyCUDA = pyCUDA_matrix_dot(A, B, WARMUP)
    results.append({"time": pyCUDA_execution_time, "result": C_pyCUDA, "comment": "pyCUDA (GPU)  Loop"})


    print("Running numba Matrix Multiplication...")
    if WARMUP:
        _ = numba_matrix_dot(A, B)
    start_time = time.perf_counter()
    C_numba = numba_matrix_dot(A, B)
    numba_execution_time = (time.perf_counter() - start_time) * 1000
    results.append({"time": numba_execution_time, "result": C_numba, "comment": "Numba JIT Loop"})


    print("Running EVE Matrix Multiplication...")
    # Call the C++ extension with EVE
    if WARMUP:
        _, _ = TEST.eve_dot2(A, B)
    # EVE_execution_time, result = TEST.eve_dot(A, B)
    # results.append({"time": EVE_execution_time, "result": result, "comment": "EVE C++ Engine"})
    EVE_execution_time, result = TEST.eve_dot2(A, B)
    results.append({"time": EVE_execution_time, "result": result, "comment": "EVE C++ Engine2"})


    print("Running KOKKOS Matrix Multiplication...")
    # Call the C++ extension with KOKKOS
    if WARMUP:
        _, _ = TEST.kokkos_dot(A, B)
    KOKKOS_execution_time, result = TEST.kokkos_dot(A, B)
    results.append({"time": KOKKOS_execution_time, "result": result, "comment": "KOKKOS C++ Engine"})


    print("Running KIWAKU Matrix Multiplication...")
    # Call the C++ extension with KIWAKU
    if WARMUP:
        _, _ = TEST.kiwaku_dot(A, B)
    KIWAKU_execution_time, result_flat = TEST.kiwaku_dot(A, B)
    results.append({"time": KIWAKU_execution_time, "result": result, "comment": "KIWAKU C++ Engine"})


    # *******************************
    #     PERFORMANCE COMPARISON
    # *******************************
    print()
    print("="*80)
    print(f" MATRIX MULTIPLICATION ({N},{M}) on ({M},{P}) PERFORMANCE COMPARISON     ")
    print("="*80)
    print(f"NumPy (np.dot) Time  : {numpy_execution_time:10.4f} ms")
    print("-" * 80)

    for result in results:
        t = result["time"]
        print(f'{result["comment"]:30s} :\t{t:10.4f} ms\t({numpy_execution_time/t:6.2f}x compare to numpy)')
        C_test = result["result"]
        matrices_match = np.allclose(C_test, C_numpy, rtol=1e-05, atol=1e-05)
        if matrices_match:
            print("\t✅ VERIFICATION SUCCESSFUL: The matrices are almost identical!")
            # Show peak difference
            max_diff = np.max(np.abs(C_test - C_numpy))
            print(f"\t\tMax absolute numerical deviation: {max_diff:.2e}")
        else:
            print("\t❌ VERIFICATION FAILED: The output matrices do not match.")
            # Diagnostics
            mismatches = np.where(~np.isclose(C_test, C_numpy, rtol=1e-05, atol=1e-05))
            print(f"\t\tNumber of elements out of tolerance: {len(mismatches[0])}")
        print("-"*80)

