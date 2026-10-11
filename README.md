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
| **CuPy (GPU) Loop** | 0.0751 ±  0.0111 ms | **9.53x** | ✅ Passed (Max Dev: 1.07e-04) |
| **pyCUDA (GPU) Loop** | 0.1114 ±  0.0050 ms | **6.42x** | ✅ Passed (Max Dev: 1.83e-04) |
| **Kokkos C++ Engine** | 0.1124 ±  0.0035 ms | **6.36x** | ✅ Passed (Max Dev: 1.83e-04) |
| **CUDA (GPU) Loop** | 0.2252 ±  0.0204 ms | **3.18x** | ✅ Passed (Max Dev: 1.83e-04) |
| **NumPy (np.dot) — Baseline** | 0.7154 ±  0.0461 ms | *1.00x* | *Reference Standard* |
| **Numba JIT Loop** | 11.3972 ±  0.4946 ms | 0.06x | ✅ Passed (Max Dev: 1.83e-04) |
| **EVE C++ Engine2** | 13.5301 ±  0.6502 ms | 0.05x | ✅ Passed (Max Dev: 1.98e-04) |
| **KIWAKU C++ Engine** | 125.1011 ±  3.3063 ms | 0.01x | ✅ Passed (Max Dev: 1.83e-04) |


### Linux Docker System Information

| Engine / Backend | Execution Time (ms) | Speedup vs. NumPy | Status Verification |
| :--- | :---: | :---: | :--- |
| **CuPy (GPU) Loop** | 0.0901 ±  0.0273 ms | **151.6x** | ✅ Passed (Max Dev: 1.07e-04) |
| **Kokkos C++ Engine** | 0.1360 ±  0.010 ms | **100.42x** | ✅ Passed (Max Dev: 1.83e-04) |
| **pyCUDA (GPU) Loop** | 0.1705 ±  0.0459 ms | **80.12x** | ✅ Passed (Max Dev: 1.83e-04) |
| **CUDA (GPU) Loop** | 0.2532 ±  0.0362 ms | **53.95x** | ✅ Passed (Max Dev: 1.83e-04) |
| **Numba JIT Loop** | 11.1702 ±  0.3146 ms | 1.22x | ✅ Passed (Max Dev: 1.83e-04) |
| **Fortran GPU Engine** | 11.3860 ±  1.5864 ms | 1.20x | ✅ Passed (Max Dev: 2.29e-04) |
| **NumPy (np.dot) — Baseline** | 5.0644 ms | *1.00x* | *Reference Standard* |
| **EVE C++ Engine2** | 14.6450 ±  0.3200 ms | 0.93x | ✅ Passed (Max Dev: 1.98e-04) |
| **KIWAKU C++ Engine** | 165.3427 ±  6.0889 ms | 0.08x | ✅ Passed (Max Dev: 1.83e-04) |
| **Fortran CPU Engine** | 211.9286 ±  4.4525 ms | 0.06x | ✅ Passed (Max Dev: 1.07e-04) |

New results 11/10/2026:
================================================================================
 MATRIX MULTIPLICATION (512,513) on (513,514) PERFORMANCE COMPARISON     
================================================================================
NumPy (np.dot) Time  :     1.3714 (median:     0.5491) ±  1.6641 ms
--------------------------------------------------------------------------------
CuPy (GPU) Loop                :            0.0955 (median:     0.0859) ±  0.0409 ms    ( 14.36x compare to numpy)
        ✅ VERIFICATION SUCCESSFUL: The matrices are almost identical!
                Max absolute numerical deviation: 1.22e-04
--------------------------------------------------------------------------------
KOKKOS C++ Engine              :            0.1276 (median:     0.1237) ±  0.0135 ms    ( 10.75x compare to numpy)
        ✅ VERIFICATION SUCCESSFUL: The matrices are almost identical!
                Max absolute numerical deviation: 2.29e-04
--------------------------------------------------------------------------------
CUDA (GPU) Loop                :            0.1346 (median:     0.1291) ±  0.0138 ms    ( 10.19x compare to numpy)
        ✅ VERIFICATION SUCCESSFUL: The matrices are almost identical!
                Max absolute numerical deviation: 2.29e-04
--------------------------------------------------------------------------------
pyCUDA (GPU) Loop              :            0.1584 (median:     0.1428) ±  0.0556 ms    (  8.66x compare to numpy)
        ✅ VERIFICATION SUCCESSFUL: The matrices are almost identical!
                Max absolute numerical deviation: 2.29e-04
--------------------------------------------------------------------------------
Fortran GPU Engine             :            2.3681 (median:     2.1709) ±  0.5997 ms    (  0.58x compare to numpy)
        ✅ VERIFICATION SUCCESSFUL: The matrices are almost identical!
                Max absolute numerical deviation: 2.29e-04
--------------------------------------------------------------------------------
Numba JIT Loop                 :           11.2062 (median:    11.1675) ±  0.4830 ms    (  0.12x compare to numpy)
        ✅ VERIFICATION SUCCESSFUL: The matrices are almost identical!
                Max absolute numerical deviation: 2.29e-04
--------------------------------------------------------------------------------
EVE C++ Engine2                :           15.0333 (median:    14.7576) ±  1.3841 ms    (  0.09x compare to numpy)
        ✅ VERIFICATION SUCCESSFUL: The matrices are almost identical!
                Max absolute numerical deviation: 2.29e-04
--------------------------------------------------------------------------------
KIWAKU C++ Engine              :          169.9970 (median:   165.9857) ±  8.9616 ms    (  0.01x compare to numpy)
        ✅ VERIFICATION SUCCESSFUL: The matrices are almost identical!
                Max absolute numerical deviation: 2.29e-04
--------------------------------------------------------------------------------
Fortran CPU Engine             :          210.5736 (median:   210.4450) ±  5.6299 ms    (  0.01x compare to numpy)
        ✅ VERIFICATION SUCCESSFUL: The matrices are almost identical!
                Max absolute numerical deviation: 2.29e-04
--------------------------------------------------------------------------------
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

