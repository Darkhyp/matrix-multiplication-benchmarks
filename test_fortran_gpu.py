import numpy as np
import hpc_benchmark_core as TEST

N = 256
A = np.ones((N, N), dtype=np.float32, order="F")
B = np.full((N, N), 2.0, dtype=np.float32, order="F")

print("Calling Fortran matmul_gpu...", flush=True)
elapsed_ms, C = TEST.matmul_gpu(A, B)

print("Returned successfully", flush=True)
print("Shape:", C.shape)
print("Elapsed:", elapsed_ms, "ms")
print("Max error:", np.max(np.abs(C - 512.0)))