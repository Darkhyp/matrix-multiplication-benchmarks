# High-Performance Matrix Multiplication Benchmark Suite

This project contains a comprehensive performance benchmarking suite designed to evaluate and compare different matrix multiplication implementation strategies across **Windows native** and **Linux Docker/WSL2 HPC environments**.

The benchmark measures raw computational throughput across:

- optimized CPU execution,
- SIMD/vectorized C++ kernels,
- vendor optimized libraries,
- CUDA GPU kernels,
- CUDA Python frameworks,
- portable GPU programming models.

The suite ensures an apples-to-apples comparison by:

- isolating pure execution time,
- avoiding asynchronous GPU measurement errors,
- synchronizing GPU pipelines,
- performing mandatory warmup runs,
- validating numerical correctness after every backend execution.

---

## 🚀 Supported Backends

### NumPy

The standard, highly optimized CPU matrix manipulation baseline utilizing continuous C-contiguous memory blocks.

### CUDA

A hand-rolled, custom C++ raw GPU execution layer interacting directly with native hardware threads.

### PyCUDA

A framework that allows for dynamic, Just-In-Time (JIT) compilation of C++ CUDA kernels directly within the Python layer.

### CuPy

An optimized, NVIDIA-native array and matrix drop-in library that mirrors the NumPy API for rapid GPU acceleration.

### EVE (Expressive Vector Engine)

A modern C++ SIMD library that leverages native vector registers (AVX2/AVX-512) to maximize per-core CPU instructions.

### Kokkos

A production-grade C++ performance portability programming model that maps data arrays (Views) and multi-dimensional parallel dispatch algorithms directly onto the GPU.

### Kiwaku

An expressive, multi-dimensional array layout engine and parallel computation wrapper written in modern C++.

---

## 💻 System Configuration & Hardware Specs

The benchmarks below were captured natively under the following hardware environment:

### System Environment Info
* **OS**: Windows 10 (v10.0.19045)
* **Architecture**: AMD64
* **CPU**: Intel64 Family 6 Model 165 Stepping 5, GenuineIntel
    * *Physical Cores*: 8 | *Logical Threads*: 16
    * *Max Frequency*: 3792.00 MHz
* **System RAM**: 31.82 GB

### Compute Device Info
* **GPU**: CUDA Available (1 Device)
* **Primary Device**: NVIDIA GeForce RTX 4090
    * *Compute Capability*: 8.9 (Ada Lovelace Architecture)
    * *Total VRAM*: 23.99 GB

---

## 📊 Benchmark Results

**Matrix Dimensions Tested**: $(512 \times 513)$ multiplied by $(513 \times 514)$

### Windows Native Environment

| Engine / Backend | Execution Time (ms) | Speedup vs. NumPy | Status Verification |
| :--- | :---: | :---: | :--- |
| **Kokkos C++ Engine** | 0.1199 ms | **6.54x** | ✅ Passed (Max Dev: 1.83e-04) |
| **pyCUDA (GPU) Loop** | 0.2701 ms | **2.90x** | ✅ Passed (Max Dev: 1.83e-04) |
| **CUDA (GPU) Loop** | 0.2931 ms | **2.68x** | ✅ Passed (Max Dev: 1.83e-04) |
| **CuPy (GPU) Loop** | 0.7149 ms | **1.10x** | ✅ Passed (Max Dev: 1.07e-04) |
| **NumPy (np.dot) — Baseline** | 0.7843 ms | *1.00x* | *Reference Standard* |
| **Numba JIT Loop** | 11.7512 ms | 0.07x | ✅ Passed (Max Dev: 1.83e-04) |
| **EVE C++ Engine2** | 15.6520 ms | 0.05x | ✅ Passed (Max Dev: 1.98e-04) |
| **KIWAKU C++ Engine** | 123.5508 ms | 0.01x | ✅ Passed (Max Dev: 1.83e-04) |


### Linux Docker System Information

| Engine / Backend | Execution Time (ms) | Speedup vs. NumPy | Status Verification |
| :--- | :---: | :---: | :--- |
| **Kokkos C++ Engine** | 0.1241 ms | **40.81x** | ✅ Passed (Max Dev: 1.83e-04) |
| **pyCUDA (GPU) Loop** | 0.1529 ms | **33.12x** | ✅ Passed (Max Dev: 1.83e-04) |
| **CUDA (GPU) Loop** | 0.2284 ms | **22.17x** | ✅ Passed (Max Dev: 1.83e-04) |
| **CuPy (GPU) Loop** | 0.7760 ms | **6.53x** | ✅ Passed (Max Dev: 1.07e-04) |
| **NumPy (np.dot) — Baseline** | 5.0644 ms | *1.00x* | *Reference Standard* |
| **Numba JIT Loop** | 11.5364 ms | 0.44x | ✅ Passed (Max Dev: 1.83e-04) |
| **EVE C++ Engine2** | 13.6215 ms | 0.37x | ✅ Passed (Max Dev: 1.98e-04) |
| **KIWAKU C++ Engine** | 161.6609 ms | 0.03x | ✅ Passed (Max Dev: 1.83e-04) |
| **Fortran CPU Engine** | 198.1114 ms | 0.01x | ✅ Passed (Max Dev: 1.07e-04) |
| **Fortran GPU Engine** | 15.8871 ms | 0.07x | ✅ Passed (Max Dev: 2.29e-04) |

Environment:

* **OS**: Linux 6.18.33.2-microsoft-standard-WSL2
* **Architecture**: x86_64
* **Python**: 3.10.12



======================================================================

---

## 🛠️ Execution & Automation Scripts

### Windows Native Environment

To streamline environment setup and execution on Windows, two specialized automation batch scripts are included in the root directory:

* **`install.bat`**: Standardizes Python environment setup. It creates/verifies your virtual environment configuration and automatically pulls down matching prerequisite binaries (`numpy`, `cupy-cuda12x`, `pycuda`, `pybind11`, `setuptools`).
* **`run.bat`**: Automated compiler pipeline tool. Cleans previous compilation fragments, triggers CMake build files for internal engines like Kokkos, compiles the `pybind11` C++ library extensions, and runs `tests.py` to produce a performance analysis report.

### Linux Docker
Located in: `test_module_linux/`

Main scripts: `install.bat`

The Linux workflow builds the complete HPC stack:

<pre style="line-height: 1.0;">
Python
  |
  ├── NumPy
  |
  ├── CuPy
  |
  ├── PyCUDA
  |
  ├── Numba
  |
  ├── pybind11
  |
  ├── CUDA
  |
  ├── Kokkos
  |
  └── NVHPC Fortran
</pre>

---

## ⚙️ Prerequisites & Manual Setup

### Windows

Required:

* Windows 10/11
* Visual Studio 2022
* Desktop C++ workload
* CUDA Toolkit
* Python 3.10+

Example MSVC path:

```plaintext
C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
```

### Linux Docker

Required:

* Linux host or WSL2
* NVIDIA GPU
* NVIDIA Container Toolkit
* Docker Compose

## 📜 License

This project is intended for research, education, and HPC benchmarking purposes.

