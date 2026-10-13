# Copyright (c) 2026
# Author: Alexandre Korovin
# e-mail: a.v.korovin73@gmail.com
# SPDX-License-Identifier: MIT

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))

import pybind11
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

import paths as PATHS

# Validation des sources C++
for source in PATHS.CPP_SOURCES:
    if not Path(source).is_file():
        raise FileNotFoundError(f"C++ source does not exist: {source}")

def needs_compilation(source_path, obj_path):
    """
    Vérifie si le fichier source est plus récent que le fichier objet (.obj/.o).
    """
    if not os.path.exists(obj_path):
        return True
    return os.path.getmtime(source_path) > os.path.getmtime(obj_path)


# ============================================================
# CONFIGURATION WINDOWS (MSVC + vcpkg OpenBLAS)
# ============================================================
if sys.platform == "win32":

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

            if not needs_compilation(src, obj_path):
                print(f"Skipping unchanged Fortran source: {src}")
                obj_files.append(obj_path)
                continue

            # Select appropriate flags depending on if it's modern .f90 or legacy .for
            flags = f90_flags if str(src).endswith(".f90") else f77_flags

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
                cmd = [compiler, "-c", str(src), "-o", obj_path] + flags

            print(f"Running: {' '.join(cmd)}")

            # Trigger actual OS process compile execution step
            subprocess.run(cmd, check=True)
            obj_files.append(obj_path)

        return obj_files


    class BuildCUDA(build_ext):

        def build_extensions(self):

            os.makedirs(self.build_temp, exist_ok=True)

            env = os.environ.copy()

            if os.path.exists(PATHS.MSVC_BIN):
                env["PATH"] = PATHS.MSVC_BIN + os.path.pathsep + env["PATH"]
            else:
                print(
                    f"⚠️ Warning: Custom MSVC path not found at: {PATHS.MSVC_BIN}. Falling back to default system path matching.")

            cuda_objects = []

            for cu in PATHS.CUDA_SOURCES:
                abs_cu_path = str((PATHS.MODULE_ROOT / cu).resolve())

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
                    str(PATHS.NVCC),
                    "-c", abs_cu_path,
                    "-o", obj,
                    "-ccbin", PATHS.MSVC_BIN,
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
                    "-Xcompiler", "/EHsc,/O2,/MT,/Zc:preprocessor,/bigobj,/permissive,/openmp",
                    "-I", str(PATHS.CUDA_INCLUDE),
                    "-I", str(PATHS.EVE_INCLUDE),
                    "-I", str(PATHS.KOKKOS_INCLUDE),
                    "-I", pybind11.get_include(),  # <-- Force Python pybind11 path injection
                    "-I", pybind11.get_include(user=True),  # <-- Include user site-packages fallback path
                    "-I", sysconfig.get_path("include")
                    # <-- Force injection of Python's own system include directory  # <-- Include user site-packages fallback path
                ]

                print("\n--- Compiling CUDA Device Code ---")
                print(" ".join(cmd))

                subprocess.check_call(cmd)
                cuda_objects.append(obj)

            for ext in self.extensions:
                ext.extra_objects.extend(cuda_objects)
                ext.include_dirs.append(str(PATHS.CUDA_INCLUDE))
                ext.library_dirs.append(str(PATHS.CUDA_LIB))
                ext.libraries.extend(["cudart"])

            super().build_extensions()


    extra_objects = []
    extra_objects += compile_fortran_sources(PATHS.FORTRAN_CPU, use_gpu=False)
    # extra_objects += compile_fortran_sources(PATHS.FORTRAN_GPU, use_gpu=True)

    extra_compile_args = [
        "/O2",
        "/EHsc",
        "/bigobj",
        "/arch:AVX2",
        '/MT',
        "/DARMA_DONT_USE_WRAPPER",
        "/DARMA_USE_BLAS",
        "/DARMA_DONT_USE_LAPACK",
    ]
    libraries = [
        "kokkoscore",
        "cudart_static",
        "cuda",
        "lapack",
        "openblas",
    ]
    extra_link_args = []
    install_requires = []
    runtime_library_dirs=[]

else:
    # ------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------

    os.environ["CC"] = "/usr/bin/gcc"
    os.environ["CXX"] = "/usr/bin/g++"

    def compile_fortran_cpu(sources, output_dir):
        """Compile les sources Fortran CPU (gfortran) en objets PIC."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        flags = ["-O3", "-fPIC", "-std=f2018"]
        objects = []

        for source in sources:
            source = Path(source)

            if not source.is_file():
                raise FileNotFoundError(f"Fortran source does not exist: {source}")

            obj = output_dir / f"{source.stem}.o"
            command = [PATHS.GFORTRAN, "-c", str(source), "-o", str(obj), *flags]

            print("\nCompiling Fortran (CPU):")
            print(" ".join(command), flush=True)

            subprocess.run(command, check=True)
            objects.append(str(obj))

        return objects


    def build_fortran_gpu_shared(sources, output_dir):
        """
        Compile + lie les sources OpenACC avec nvfortran en bibliotheque
        partagee. C'est nvfortran qui fait l'etape de device-link (nvlink)
        et qui enregistre le binaire GPU aupres du runtime OpenACC.
        """

        if not PATHS.NVFORTRAN.is_file():
            raise RuntimeError(f"nvfortran not found: {PATHS.NVFORTRAN}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for source in sources:
            if not Path(source).is_file():
                raise FileNotFoundError(f"Fortran source does not exist: {source}")

        lib = output_dir / PATHS.GPU_LIB_FILE

        command = [
            str(PATHS.NVFORTRAN),
            "-O3",
            "-fPIC",
            "-shared",
            "-acc=gpu",
            "-gpu=cuda12.4,cc89",
            *map(str, sources),
            "-o", str(lib),
            f"-Wl,-rpath,{PATHS.NVHPC_LIB}",
            f"-Wl,-rpath,{PATHS.CUDA_LIB}",
        ]

        print("\nBuilding Fortran GPU shared library:")
        print(" ".join(command), flush=True)

        subprocess.run(command, check=True)
        return output_dir, lib


    # ------------------------------------------------------------
    # C++ / CUDA extension
    # ------------------------------------------------------------

    class BuildCUDA(build_ext):

        def build_extensions(self):
            build_temp = Path(self.build_temp).resolve()
            build_temp.mkdir(parents=True, exist_ok=True)

            # --- Fortran CPU (objets) ---
            fortran_objects = compile_fortran_cpu(
                PATHS.FORTRAN_CPU,
                build_temp / "fortran_cpu",
            )

            # --- Fortran GPU (lib partagee device-linkee) ---
            gpu_lib_dir, gpu_lib = build_fortran_gpu_shared(
                PATHS.FORTRAN_GPU,
                build_temp / "fortran_gpu",
            )
            self._gpu_lib = gpu_lib

            cuda_objects = []

            for source in PATHS.CUDA_SOURCES:
                if not source.is_file():
                    raise FileNotFoundError(f"CUDA source does not exist: {source}")

                obj = build_temp / f"{source.stem}.o"

                command = [
                    str(PATHS.NVCC),
                    "-c", str(source),
                    "-o", str(obj),
                    "-std=c++20",
                    "--extended-lambda",
                    "-O3",
                    "-Xcompiler", "-fPIC",
                    "-I", str(PATHS.CUDA_INCLUDE),
                    "-I", str(PATHS.KOKKOS_INCLUDE),
                    "-I", pybind11.get_include(),
                    "-I", sysconfig.get_path("include"),
                    "-gencode=arch=compute_89,code=sm_89",
                ]

                print("\nCompiling CUDA:")
                print(" ".join(command), flush=True)

                subprocess.run(command, check=True)
                cuda_objects.append(str(obj))

            # --- Branchement sur l'extension ---
            for ext in self.extensions:
                ext.extra_objects.extend(fortran_objects)
                ext.extra_objects.extend(cuda_objects)

                ext.library_dirs.append(str(gpu_lib_dir))
                if PATHS.GPU_LIB_NAME not in ext.libraries:
                    ext.libraries.append(PATHS.GPU_LIB_NAME)

                # $ORIGIN : la lib est copiee a cote du .so Python.
                # gpu_lib_dir : repli pour les builds en place / debug.
                for rp in ("$ORIGIN", str(PATHS.MODULE_ROOT)):
                    if rp not in ext.runtime_library_dirs:
                        ext.runtime_library_dirs.append(rp)

            super().build_extensions()

            # Copie libmatmul_acc.so à côté de l'extension construite
            # ET dans MODULE_ROOT (stable, survit au build temporaire de pip -e).
            for ext in self.extensions:
                dest_dirs = {
                    Path(self.get_ext_fullpath(ext.name)).resolve().parent,
                    PATHS.MODULE_ROOT.resolve(),
                }
                for dest_dir in dest_dirs:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / PATHS.GPU_LIB_FILE
                    print(f"\nCopying {gpu_lib} -> {dest}", flush=True)
                    shutil.copy2(gpu_lib, dest)


    CPP_SOURCES = [
        os.path.relpath(s, start=HERE).replace("\\", "/")
        for s in PATHS.CPP_SOURCES
    ]

    extra_compile_args = [
        "-O3",
        "-fPIC",
        "-mavx2",
        "-fopenmp",
        "-DHAS_FORTRAN"
    ]
    extra_link_args = [
        "-Wl,--no-as-needed",
        f"-Wl,-rpath,{PATHS.NVHPC_LIB}",
        "-lgomp"  # Pour OpenMP sous GCC
    ]
    libraries = [
        "kokkoscore",
        "cudart",
        "cuda",
        "openblas",
        "gfortran",
        "quadmath",
        "dl",
        "pthread",
        "m",
    ]
    install_requires = [
        "numpy",
        "cupy-cuda12x",
        "numba",
        "pycuda",
        "psutil",
    ]
    runtime_library_dirs=[
        str(PATHS.NVHPC_LIB),
        str(PATHS.CUDA_LIB),
    ]
    extra_objects = []

# ------------------------------------------------------------
# Extension sources
# ------------------------------------------------------------

ext_modules = [
    Pybind11Extension(
        "hpc_benchmark_core",
        PATHS.CPP_SOURCES,
        include_dirs=PATHS.INCLUDE_DIRS,
        library_dirs=PATHS.LIBRARY_DIRS,
        libraries=libraries,
        cxx_std=20,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        extra_objects=extra_objects,
        runtime_library_dirs=runtime_library_dirs,
    ),
]

# ------------------------------------------------------------
# Package
# ------------------------------------------------------------

setup(
    name="hpc_test_extensions",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildCUDA},
    install_requires=install_requires,
)
