import os
import subprocess
import sys
import sysconfig
import pybind11
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
# import shutil
#
# shutil.rmtree("build", ignore_errors=True)

# Force GCC for Python extension compilation
os.environ["CC"] = "/usr/bin/gcc"
os.environ["CXX"] = "/usr/bin/g++"

# Native Linux Path Configurations
KOKKOS_DIR = "/workspace/test_module/KOKKOS_SRC/dist_linux"
CUDA_INCLUDE = "/usr/local/cuda/include"
CUDA_LIB_DIR = "/usr/local/cuda/lib64"
NVCC = "/usr/local/cuda/bin/nvcc"

NVHPC_ROOT = "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5"
NVHPC_LIB = f"{NVHPC_ROOT}/compilers/lib"

# Adjusted header dependencies (Looking into parent directory workspace)
EVE_INCLUDE = os.path.abspath("/workspace/test_module/EVE_SRC/include")  # Adjust if paths differ
KIWAKU_INCLUDE = os.path.abspath("/workspace/test_module/KIWAKU_SRC/include")  # Adjust if paths differ
RABERU_INCLUDE = os.path.abspath("/workspace/test_module/RABERU_SRC/include")
KUMI_INCLUDE = os.path.abspath("/workspace/test_module/KUMI_SRC/include")

# Change the relative paths to absolute workspace targets:
# fortran_sources_CPU = ["../src/FORTRAN/matrix_dot.f90"]
# fortran_sources_GPU = ["../src/FORTRAN/matrix_dot2.f90"]
fortran_sources_CPU = ["/workspace/test_module/src/FORTRAN/matrix_dot.f90"]
fortran_sources_GPU = ["/workspace/test_module/src/FORTRAN/matrix_dot2.f90"]

def compile_fortran_sources(fortran_sources, use_gpu=False):
    """Manually compiles Fortran sources into Linux .o files natively inside the container."""
    if use_gpu:
        print("--- Compiling Fortran via NVIDIA nvfortran (GPU Offload) ---")
        compiler = "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/bin/nvfortran"
        flags = ["-O3", "-stdpar=gpu", "-gpu=cuda12.4,cc89", "-fPIC"]

        if not os.path.exists(compiler):
            print("⚠️ nvfortran not found at standard path! Falling back to gfortran-12 for GPU simulation layer.")
            compiler = "gfortran-12"  # 👈 Changed to gfortran-12
            flags = ["-O3", "-fopenmp", "-fPIC", "-std=f2018"]
    else:
        print("--- Compiling Fortran via GCC gfortran (CPU + OpenBLAS) ---")
        compiler = "gfortran-12"  # 👈 Changed to gfortran-12
        flags = ["-O3", "-march=native", "-ffast-math", "-fPIC", "-std=f2018"]

    obj_files = []
    out_dir = os.path.abspath("build/fortran_objs")
    os.makedirs(out_dir, exist_ok=True)

    for src in fortran_sources:
        if not os.path.exists(src):
            print(f"❌ Fortran Source missing: {src}")
            continue

        base_name = os.path.basename(src)
        obj_name = os.path.splitext(base_name)[0] + ".o"  # Linux uses .o
        obj_path = os.path.join(out_dir, obj_name)

        cmd = [compiler, "-c", src, "-o", obj_path] + flags
        print(f"Running: {' '.join(cmd)}")
        # subprocess.run(cmd, check=True)
        subprocess.check_call(cmd)
        obj_files.append(obj_path)

    return obj_files


class BuildCUDALinux(build_ext):
    def build_extensions(self):
        os.makedirs(self.build_temp, exist_ok=True)
        cu_sources = [
            # "../src/CUDA/matrix_dot_kernel.cu",
            # "../src/KOKKOS/matrix_dot.cu",
            "/workspace/test_module/src/CUDA/matrix_dot_kernel.cu",
            "/workspace/test_module/src/KOKKOS/matrix_dot.cu",
        ]
        cuda_objects = []

        for cu in cu_sources:
            abs_cu_path = os.path.abspath(cu)
            print(f"🔍 Checking CUDA source: {abs_cu_path}")
            if not os.path.exists(abs_cu_path):
                print(f"❌ ERROR: File does not exist: {abs_cu_path}")
                sys.exit(1)

            obj = os.path.join(
                self.build_temp,
                os.path.basename(cu).replace(".cu", ".o")
            )
            os.makedirs(os.path.dirname(obj), exist_ok=True)

            # Linux GCC friendly nvcc invocation flags
            cmd = [
                NVCC, "-c", abs_cu_path, "-o", obj,
                "-maxrregcount=0", "--machine", "64", "-cudart", "shared",
                "-std=c++20", "--generate-code=arch=compute_89,code=[compute_89,sm_89]",
                "-extended-lambda", "-Wext-lambda-captures-this", "-O3", "--use_fast_math",
                "-Xcompiler", "-fPIC,-O3",  # Linux GCC shared library layout argument flags
                "-I", CUDA_INCLUDE,
                "-I", EVE_INCLUDE,
                "-I", os.path.join(KOKKOS_DIR, "include"),
                "-I", pybind11.get_include(),
                "-I", pybind11.get_include(user=True),
                "-I", sysconfig.get_path("include")
            ]

            print(f"\n--- Compiling CUDA Device Code Natively ---\n{' '.join(cmd)}")
            subprocess.check_call(cmd)
            cuda_objects.append(obj)

        for ext in self.extensions:
            ext.extra_objects.extend(cuda_objects)
            ext.include_dirs.append(CUDA_INCLUDE)
            ext.library_dirs.append(CUDA_LIB_DIR)
            ext.libraries.extend(["cudart", "cuda"])

        super().build_extensions()


# Compile manual dependencies
compiled_fortran_objects_CPU = compile_fortran_sources(fortran_sources_CPU, use_gpu=False)
compiled_fortran_objects_GPU = compile_fortran_sources(fortran_sources_GPU, use_gpu=True)

# Combine objects into a clean linear list array
all_extra_objects = compiled_fortran_objects_CPU + compiled_fortran_objects_GPU

ext_modules = [
    Pybind11Extension(
        "hpc_benchmark_core",
        [
            # "../src/KIWAKU/matrix_dot.cpp",
            # "../src/binding.cpp",
            # "../src/EVE/matrix_dot.cpp",
            # "../src/EVE/matrix_dot2.cpp",
            # "../src/CUDA/matrix_op.cpp",
            "/workspace/test_module/src/binding_linux.cpp",
            "/workspace/test_module/src/KIWAKU/matrix_dot.cpp",
            "/workspace/test_module/src/EVE/matrix_dot.cpp",
            "/workspace/test_module/src/EVE/matrix_dot2.cpp",
            "/workspace/test_module/src/CUDA/matrix_op.cpp",
        ],
        include_dirs=[
            EVE_INCLUDE, KIWAKU_INCLUDE, RABERU_INCLUDE, KUMI_INCLUDE,
            os.path.join(KOKKOS_DIR, "include"),
            "/workspace/test_module/src"
        ],
        library_dirs=[
            os.path.join(KOKKOS_DIR, "lib"),
            os.path.join(KOKKOS_DIR, "lib64"),
            CUDA_LIB_DIR,
            NVHPC_LIB,
        ],
        libraries=[
            "kokkoscore",
            # CUDA
            "cudart",
            "cuda",
            # BLAS
            "openblas",
            # GNU Fortran runtime
            "gfortran",
            "quadmath",
            # NVIDIA Fortran runtime
            "nvf",
            "nvomp",
            "nvhpcatm",
            "nvcpumath",
            "nsnvc",
            "nvc",
            "dl",
            "pthread",
            "m",
        ],
        cxx_std=20,
        extra_compile_args=[
            "-std=c++20",
            "-O3",
            "-fPIC",
            "-mavx2"
        ],
        extra_objects=all_extra_objects,
        extra_link_args=[
            "-Wl,--no-as-needed",
            "-L" + NVHPC_LIB,
            "-Wl,-rpath," + NVHPC_LIB,
        ],
        runtime_library_dirs=[
            NVHPC_LIB,
        ],
    ),
]

setup(
    name="hpc_test_extensions",
    version="1.0",
    install_requires=[
        "numpy",
        "cupy-cuda12x",
        "numba",
        "pycuda",
        "psutil",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildCUDALinux},
)