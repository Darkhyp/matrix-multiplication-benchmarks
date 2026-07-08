import platform
import time

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes the CUDA context
from pycuda.compiler import SourceModule

if platform.system() == "Linux":
    print("Running on Linux")
    nvcc_options = [
        "-ccbin", "/usr/bin/g++",
    ]
else:
    from test_module.pathes import MSVC_BIN_DIR
    nvcc_options = [
        "-ccbin", MSVC_BIN_DIR,
        "-allow-unsupported-compiler",
    ]

# the raw C++ CUDA kernel
mod = SourceModule("""
    __global__ void pycuda_matmul(const float *A, const float *B, float *C, 
                                  int N, int M, int P) 
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < N && col < P) {
            float sum = 0.0f;
            for (int k = 0; k < M; ++k) {
                sum += A[row * M + k] * B[k * P + col];
            }
            C[row * P + col] = sum;
        }
    }
    """,
    options=nvcc_options
)

def pyCUDA_matrix_dot(A, B, WARMUP=False):
    pycuda.autoinit.context.push()

    rows_A, cols_A, cols_B = A.shape[0], A.shape[1], B.shape[1]

    try:
        # Get the compiled kernel function reference
        matmul_kernel = mod.get_function("pycuda_matmul")

        # 2. Allocate VRAM memory on the GPU
        A_gpu_pycuda = cuda.mem_alloc(A.nbytes)
        B_gpu_pycuda = cuda.mem_alloc(B.nbytes)
        C_gpu_pycuda = cuda.mem_alloc(rows_A * cols_B * A.itemsize)

        # Copy matrices from CPU (Host) to GPU (Device)
        cuda.memcpy_htod(A_gpu_pycuda, A)
        cuda.memcpy_htod(B_gpu_pycuda, B)

        # Calculate grid/block thread configurations
        block_dim = (16, 16, 1)
        grid_dim = (int(np.ceil(cols_B / 16)), int(np.ceil(rows_A / 16)), 1)

        # --- Warmup Call ---
        # Run once to eliminate any initialization or first-run driver lag
        if WARMUP:
            matmul_kernel(A_gpu_pycuda, B_gpu_pycuda, C_gpu_pycuda,
                          np.int32(rows_A), np.int32(cols_A), np.int32(cols_B),
                          block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()  # Ensure warmup execution finishes

        # Pure Execution Benchmark
        start_time = time.perf_counter()

        matmul_kernel(A_gpu_pycuda, B_gpu_pycuda, C_gpu_pycuda,
                      np.int32(rows_A), np.int32(cols_A), np.int32(cols_B),
                      block=block_dim, grid=grid_dim)

        # CRITICAL: Force the CPU to wait until the PyCUDA stream is completely empty
        cuda.Context.synchronize()

        pycuda_execution_time = (time.perf_counter() - start_time) * 1000

        # Fetch output matrix back to host memory for verification
        C_pycuda = np.empty((rows_A, cols_B), dtype=np.float32)
        cuda.memcpy_dtoh(C_pycuda, C_gpu_pycuda)

        # Free memory explicitly (optional but good practice)
        A_gpu_pycuda.free()
        B_gpu_pycuda.free()
        C_gpu_pycuda.free()

        return pycuda_execution_time, C_pycuda

    finally:
        # Cleanly pop the context stack so CuPy can resume later if needed
        pycuda.autoinit.context.pop()