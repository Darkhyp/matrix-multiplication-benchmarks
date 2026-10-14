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

| Engine / Backend      | Execution Time (ms)  | Speedup vs. NumPy | Status Verification           |
|:----------------------|:--------------------:|:-----------------:|:------------------------------|
| **CuPy (GPU) Loop**   |   0.0751 ±  0.0111   |     **9.53x**     | ✅ Passed (Max Dev: 1.07e-04) |
| **pyCUDA (GPU) Loop** |   0.1114 ±  0.0050   |     **6.42x**     | ✅ Passed (Max Dev: 1.83e-04) |
| **Kokkos C++ Engine** |   0.1124 ±  0.0035   |     **6.36x**     | ✅ Passed (Max Dev: 1.83e-04) |
| **CUDA (GPU) Loop**   |   0.2252 ±  0.0204   |     **3.18x**     | ✅ Passed (Max Dev: 1.83e-04) |
| **NumPy (np.dot)**    |  0.7154 ±  0.0461    |   *1.00x*         | *Reference Standard*          |
| **Numba JIT Loop**    |  11.3972 ±  0.4946   |       0.06x       | ✅ Passed (Max Dev: 1.83e-04) |
| **EVE C++ Engine2**   |  13.5301 ±  0.6502   |       0.05x       | ✅ Passed (Max Dev: 1.98e-04) |
| **KIWAKU C++ Engine** |  125.1011 ±  3.3063  |       0.01x       | ✅ Passed (Max Dev: 1.83e-04) |


### Linux Docker System Information

| Engine / Backend       | Execution Time (ms)  | Speedup vs. NumPy | Status Verification          |
|:-----------------------|:--------------------:|:-----------------:|:-----------------------------|
| **CuPy (GPU) Loop**    |   0.0901 ±  0.0273   |    **151.6x**     | ✅ Passed (Max Dev: 1.07e-04) |
| **Kokkos C++ Engine**  |   0.1360 ±  0.010    |    **100.42x**    | ✅ Passed (Max Dev: 1.83e-04) |
| **pyCUDA (GPU) Loop**  |   0.1705 ±  0.0459   |    **80.12x**     | ✅ Passed (Max Dev: 1.83e-04) |
| **CUDA (GPU) Loop**    |   0.2532 ±  0.0362   |    **53.95x**     | ✅ Passed (Max Dev: 1.83e-04) |
| **Numba JIT Loop**     |  11.1702 ±  0.3146   |       1.22x       | ✅ Passed (Max Dev: 1.83e-04) |
| **Fortran GPU Engine** |  11.3860 ±  1.5864   |       1.20x       | ✅ Passed (Max Dev: 2.29e-04) |
| **NumPy (np.dot)**     |    5.0644            |   *1.00x*         | *Reference Standard*         |
| **EVE C++ Engine2**    |  14.6450 ±  0.3200   |       0.93x       | ✅ Passed (Max Dev: 1.98e-04) |
| **KIWAKU C++ Engine**  |  165.3427 ±  6.0889  |       0.08x       | ✅ Passed (Max Dev: 1.83e-04) |
| **Fortran CPU Engine** |  211.9286 ±  4.4525  |       0.06x       | ✅ Passed (Max Dev: 1.07e-04) |

### New results (Linux Docker) 14/10/2026

| Engine / Backend               | Execution Time (ms) | median (ms) | Speedup vs. NumPy | Status Verification      |
|:-------------------------------|:-------------------:|:-----------:|:-----------------:|:-------------------------|
| **cuBLAS GPU C++ Engine**      |     0.0486 ± 0.0036 |      0.0475 |   12.61(   8.79)x | ✅ Passed (Max Dev: 1.07e-04) |
| **CuPy (GPU) Loop**            |     0.0906 ± 0.0303 |      0.0797 |    6.76(   5.24)x | ✅ Passed (Max Dev: 1.07e-04) |
| **CUDA cu (GPU) Loop2**        |     0.1139 ± 0.0059 |      0.1114 |    5.38(   3.75)x | ✅ Passed (Max Dev: 2.59e-04) |
| **KOKKOS C++ Engine**          |     0.1239 ± 0.0085 |      0.1213 |    4.95(   3.45)x | ✅ Passed (Max Dev: 2.59e-04) |
| **pyCUDA (GPU) Loop**          |     0.1256 ± 0.0088 |      0.1219 |    4.88(   3.43)x | ✅ Passed (Max Dev: 2.59e-04) |
| **NumPy (np.dot)**             |     0.6127 ± 0.5572 |      0.4179 | *1.00x*           | *Reference Standard*         |
| **ARMADILLO CPU C++ Engine**   |     0.8915 ± 0.6500 |      0.8644 |    0.69(   0.48)x | ✅ Passed (Max Dev: 1.37e-04) |
| **EIGEN C++ Engine**           |     0.9754 ± 0.3953 |      0.8716 |    0.63(   0.48)x | ✅ Passed (Max Dev: 1.37e-04) |
| **KIWAKU C++ Engine2**         |     1.7733 ± 0.7382 |      1.4759 |    0.35(   0.28)x | ✅ Passed (Max Dev: 2.59e-04) |
| **Fortran GPU Engine**         |     2.1994 ± 0.5319 |      2.0720 |    0.28(   0.20)x | ✅ Passed (Max Dev: 2.59e-04) |
| **EVE C++ Engine2**            |     2.4391 ± 0.8486 |      2.5567 |    0.25(   0.16)x | ✅ Passed (Max Dev: 2.59e-04) |
| **Numba JIT Loop**             |    10.9376 ± 0.3295 |     10.9008 |    0.06(   0.04)x | ✅ Passed (Max Dev: 2.59e-04) |
| **Fortran CPU Engine**         |   210.8646 ± 7.0287 |    210.1303 |    0.00(   0.00)x | ✅ Passed (Max Dev: 2.59e-04) |


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
* **`make_module.bat`**: Automated compiler pipeline tool. Cleans previous compilation fragments, triggers CMake build files for internal engines like Kokkos, compiles the `pybind11` C++ library extensions, and runs `tests.py` to produce a performance analysis report.

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

