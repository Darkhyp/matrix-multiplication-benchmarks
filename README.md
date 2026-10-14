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

| Engine / Backend      |  Execution Time (ms)  | Speedup vs. NumPy | Status Verification           |
|:----------------------|:---------------------:|:-----------------:|:------------------------------|
| **CuPy (GPU) Loop**   |  0.0751 ±  0.0111   |     **9.53x**     | ✅ Passed (Max Dev: 1.07e-04) |
| **pyCUDA (GPU) Loop** |  0.1114 ±  0.0050   |     **6.42x**     | ✅ Passed (Max Dev: 1.83e-04) |
| **Kokkos C++ Engine** |  0.1124 ±  0.0035   |     **6.36x**     | ✅ Passed (Max Dev: 1.83e-04) |
| **CUDA (GPU) Loop**   |  0.2252 ±  0.0204   |     **3.18x**     | ✅ Passed (Max Dev: 1.83e-04) |
| **NumPy (np.dot)**    |  0.7154 ±  0.0461   |      *1.00x*      | *Reference Standard*          |
| **Numba JIT Loop**    | 11.3972 ±  0.4946   |       0.06x       | ✅ Passed (Max Dev: 1.83e-04) |
| **EVE C++ Engine2**   | 13.5301 ±  0.6502   |       0.05x       | ✅ Passed (Max Dev: 1.98e-04) |
| **KIWAKU C++ Engine** | 125.1011 ±  3.3063  |       0.01x       | ✅ Passed (Max Dev: 1.83e-04) |


### Linux Docker System Information

| Engine / Backend       |  Execution Time (ms)  | Speedup vs. NumPy | Status Verification            |
|:-----------------------|:---------------------:|:-----------------:|:-------------------------------|
| **CuPy (GPU) Loop**    | 0.0901 ±  0.0273    |   **151.6x**      | ✅ Passed (Max Dev: 1.07e-04)  |
| **Kokkos C++ Engine**  |  0.1360 ±  0.010    |    **100.42x**    | ✅ Passed (Max Dev: 1.83e-04)  |
| **pyCUDA (GPU) Loop**  |  0.1705 ±  0.0459   |    **80.12x**     | ✅ Passed (Max Dev: 1.83e-04)  |
| **CUDA (GPU) Loop**    |  0.2532 ±  0.0362   |    **53.95x**     | ✅ Passed (Max Dev: 1.83e-04)  |
| **Numba JIT Loop**     | 11.1702 ±  0.3146   |       1.22x       | ✅ Passed (Max Dev: 1.83e-04)  |
| **Fortran GPU Engine** | 11.3860 ±  1.5864   |       1.20x       | ✅ Passed (Max Dev: 2.29e-04)  |
| **NumPy (np.dot)**     |       5.0644        |      *1.00x*      | *Reference Standard*           |
| **EVE C++ Engine2**    | 14.6450 ±  0.3200   |       0.93x       | ✅ Passed (Max Dev: 1.98e-04)  |
| **KIWAKU C++ Engine**  | 165.3427 ±  6.0889  |       0.08x       | ✅ Passed (Max Dev: 1.83e-04)  |
| **Fortran CPU Engine** | 211.9286 ±  4.4525  |       0.06x       | ✅ Passed (Max Dev: 1.07e-04)  |

### New results (Linux Docker) 13/10/2026

| Engine / Backend             | Execution Time (ms) | median (ms) | Speedup vs. NumPy | Status Verification         |
|:-----------------------------|:-------------------:|:-----------:|:-----------------:|:----------------------------|
| **ARMADILLO C++ Engine**     |   0.0492 ± 0.0063   |   0.0474    |   55.97(41.10)x   | ✅ Passed (Max Dev: 1.22e-04 |
| **CuPy (GPU) Loop**          |   0.0864 ± 0.0171   |   0.0801    |   31.85(24.31)x   | ✅ Passed (Max Dev: 1.22e-04 |
| **CUDA cu (GPU) Loop2**      |   0.1147 ± 0.0050   |   0.1135    |   24.00(17.16)x   | ✅ Passed (Max Dev: 2.29e-04 |
| **KOKKOS C++ Engine**        |   0.1230 ± 0.0032   |   0.1218    |   22.39(15.99)x   | ✅ Passed (Max Dev: 2.29e-04 |
| **pyCUDA (GPU) Loop**        |   0.1247 ± 0.0052   |   0.1242    |   22.09(15.68)x   | ✅ Passed (Max Dev: 2.29e-04 |
| **EIGEN C++ Engine**         |   0.5821 ± 0.2654   |   0.5171    |    4.73(3.77)x    | ✅ Passed (Max Dev: 1.22e-04 |
| **ARMADILLO CPU C++ Engine** |   0.5854 ± 0.1562   |   0.5260    |    4.70(3.70)x    | ✅ Passed (Max Dev: 1.22e-04 |
| **EVE C++ Engine2**          |   2.5464 ± 1.6192   |   1.5986    |    1.08(1.22)x    | ✅ Passed (Max Dev: 2.14e-04 |
| **KIWAKU C++ Engine2**       |   3.1506 ± 1.8382   |   2.8169    |    0.87(0.69)x    | ✅ Passed (Max Dev: 2.14e-04 |
| **Fortran GPU Engine**       |   1.9698 ± 0.5044   |   1.8396    |    1.40(1.06)x    | ✅ Passed (Max Dev: 2.29e-04 |
| **NumPy (np.dot)**           |  2.7534 ± 2.8899    | 1.9471      | *1.00x*           | *Reference Standard*        |
| **Numba JIT Loop**           |  10.8720 ± 0.6165   |   10.8171   |    0.25(0.18)x    | ✅ Passed (Max Dev: 2.29e-04 |
| **Fortran CPU Engine**       |  211.1064 ± 6.2598  |  212.2016   |    0.01(0.01)x    | ✅ Passed (Max Dev: 2.14e-04 |

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

