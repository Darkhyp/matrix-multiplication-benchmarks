import os
import subprocess
import sys
import sysconfig

import pybind11
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

from pathes import CUDA_INCLUDE, NVCC, MSVC_BIN_DIR, current_dir, EVE_INCLUDE, KOKKOS_INCLUDE, KOKKOS_LIB_DIR, \
    CUDA_LIB_DIR, KIWAKU_INCLUDE, RABERU_INCLUDE, KUMI_INCLUDE, MSYS_LIB_DIR

fortran_sources_CPU = [
    "src/FORTRAN/matrix_dot.f90",
]
fortran_sources_GPU = [
    "src/FORTRAN/matrix_dot2.f90",
]


def compile_fortran_sources(fortran_sources, use_gpu=False):
    """
    Manually compiles Fortran sources into .obj files before pybind11 builds.
    Returns a list of paths to the compiled object files.
    """
    # Choose compiler based on target choice
    if use_gpu:
        print("--- Compiling Fortran via NVIDIA nvfortran (GPU Offload) ---")

        compiler = "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/bin/nvfortran"

        # FIX: Explicitly add 'cuda12.4,' right before cc89
        f90_flags = ["-O3", "-stdpar=gpu", "-gpu=cuda12.4,cc89"]
        f77_flags = ["-O3", "-stdpar=gpu", "-gpu=cuda12.4,cc89", "-Mfixed"]
    else:
        print("--- Compiling Fortran via GCC gfortran (CPU + OpenBLAS) ---")
        compiler = "gfortran"
        f90_flags = ["-O3", "-march=native", "-ffast-math"]
        f77_flags = ["-O3", "-march=native", "-ffast-math", "-ffixed-line-length-none"]

    obj_files = []
    out_dir = os.path.abspath("build/fortran_objs")
    os.makedirs(out_dir, exist_ok=True)

    for src in fortran_sources:
        base_name = os.path.basename(src)
        obj_name = os.path.splitext(base_name)[0] + ".obj"
        obj_path = os.path.join(out_dir, obj_name)

        # Select appropriate flags depending on if it's modern .f90 or legacy .for
        flags = f90_flags if src.endswith(".f90") else f77_flags

        # Build command: compiler -c source_file -o object_file flags...
        # Fix paths for WSL if using GPU, otherwise use native Windows paths
        if use_gpu:
            # Convert Windows paths (F:\path\to\file) to WSL paths (/mnt/f/path/to/file)
            abs_src = os.path.abspath(src)
            wsl_src = abs_src.replace("\\", "/").replace(":", "")
            wsl_src = f"/mnt/{wsl_src[0].lower()}{wsl_src[1:]}"

            wsl_obj = obj_path.replace("\\", "/").replace(":", "")
            wsl_obj = f"/mnt/{wsl_obj[0].lower()}{wsl_obj[1:]}"

            # Join flags into a single space-separated string for bash
            flags_str = " ".join(flags)

            # FIX: Use explicit export statements and wrap values in quotes
            # to protect spaces and parentheses from breaking bash parsing logic
            bash_command = (
                'export PATH="/usr/lib/wsl/lib:$PATH"; '
                'export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/12.4/lib64:$LD_LIBRARY_PATH"; '
                f'"{compiler}" -c "{wsl_src}" -o "{wsl_obj}" {flags_str}'
            )

            cmd = ["wsl", "bash", "-c", bash_command]
        else:
            # Standard local windows gfortran execution
            cmd = [compiler, "-c", src, "-o", obj_path] + flags

        print(f"Running: {' '.join(cmd)}")

        # Trigger actual OS process compile execution step
        subprocess.run(cmd, check=True)
        obj_files.append(obj_path)

    return obj_files


class BuildCUDA(build_ext):

    def build_extensions(self):

        os.makedirs(self.build_temp, exist_ok=True)

        env = os.environ.copy()

        if os.path.exists(MSVC_BIN_DIR):
            env["PATH"] = MSVC_BIN_DIR + os.path.pathsep + env["PATH"]
        else:
            print(
                f"⚠️ Warning: Custom MSVC path not found at: {MSVC_BIN_DIR}. Falling back to default system path matching.")

        cu_sources = [
            "src/CUDA/matrix_dot_kernel.cu",
            "src/KOKKOS/matrix_dot.cu",
        ]

        cuda_objects = []

        for cu in cu_sources:
            abs_cu_path = os.path.abspath(os.path.join(current_dir, cu))

            # --- CRITICAL SANITY CHECK ---
            print(f"\n🔍 Python checking path: {abs_cu_path}")
            if not os.path.exists(abs_cu_path):
                print(f"❌ ERROR: File does not exist at that path!")
                print("Please check your file extensions (.cu vs .cpp) and capitalization!")
                sys.exit(1)
            else:
                print("✅ File verified on disk by Python.")
            # -----------------------------

            obj = os.path.join(
                self.build_temp,
                os.path.dirname(cu),
                os.path.basename(cu).replace(".cu", ".obj")
            )
            print("build_temp =", repr(self.build_temp))
            print("build_lib =", repr(self.build_lib))
            print("obj =", repr(obj))

            os.makedirs(os.path.dirname(obj), exist_ok=True)

            cmd = [
                NVCC,
                "-c",
                abs_cu_path,
                "-o",
                obj,
                "-ccbin", MSVC_BIN_DIR,
                "-maxrregcount=0",
                "--machine", "64",
                "-cudart", "static",
                "-allow-unsupported-compiler",
                "-std=c++20",
                "--generate-code=arch=compute_89,code=[compute_89,sm_89]",
                "-extended-lambda",
                "-Wext-lambda-captures-this",
                "-O3",
                "--use_fast_math",
                "-Xcompiler", "/EHsc,/O2,/MD,/Zc:preprocessor,/bigobj,/permissive",
                "-I", CUDA_INCLUDE,
                "-I", EVE_INCLUDE,
                "-I", KOKKOS_INCLUDE,
                "-I", pybind11.get_include(),  # <-- Force Python pybind11 path injection
                "-I", pybind11.get_include(user=True),  # <-- Include user site-packages fallback path
                "-I", sysconfig.get_path("include") # <-- Force injection of Python's own system include directory  # <-- Include user site-packages fallback path
            ]

            print("\n--- Compiling CUDA Device Code ---")
            print(" ".join(cmd))

            subprocess.check_call(cmd)
            cuda_objects.append(obj)

        for ext in self.extensions:
            ext.extra_objects.extend(cuda_objects)
            ext.include_dirs.append(CUDA_INCLUDE)
            ext.library_dirs.append(CUDA_LIB_DIR)
            ext.libraries.extend(["cudart"])

        super().build_extensions()


compiled_fortran_objects_CPU = compile_fortran_sources(fortran_sources_CPU, use_gpu=False)
compiled_fortran_objects_GPU = compile_fortran_sources(fortran_sources_GPU, use_gpu=True)

ext_modules = [
    Pybind11Extension(
        "test_module",
        [
            "src/KIWAKU/matrix_dot.cpp",
            # "src/KOKKOS/matrix_dot.cpp",
            "src/binding.cpp",
            "src/EVE/matrix_dot.cpp",
            "src/EVE/matrix_dot2.cpp",
            "src/CUDA/matrix_op.cpp",
        ],
        include_dirs=[
            EVE_INCLUDE,
            KIWAKU_INCLUDE,
            RABERU_INCLUDE,
            KUMI_INCLUDE,
            KOKKOS_INCLUDE,
            "src"
        ],
        library_dirs=[
            KOKKOS_LIB_DIR,
            CUDA_LIB_DIR,
        ],
        libraries=[
            "kokkoscore",
            "cudart",
            "cuda"
        ],
        extra_compile_args=[
            "/std:c++20",
            "/O2",
            "/EHsc",
            '/bigobj',
            "/arch:AVX2", # Adjust /arch if using AVX512 or NEON
        ],
        extra_link_args=[
            "cuda.lib"
        ],
        # extra_objects=[
        #     compiled_fortran_objects_CPU,
        #     compiled_fortran_objects_GPU,
        #     os.path.join(MSYS_LIB_DIR, "libopenblas.dll.a")
        # ],
    ),
]

setup(
    name="test_module",
    version="1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildCUDA},
)