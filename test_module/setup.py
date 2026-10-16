# Copyright (c) 2026
# Author: Alexandre Korovin
# e-mail: a.v.korovin73@gmail.com
# SPDX-License-Identifier: MIT

"""
Build de l'extension `hpc_benchmark_core` (pybind11 + CUDA + Kokkos + Fortran).

Variables d'environnement optionnelles :
    CUDA_ARCH=89        architecture GPU (sm_XX)
    CUDA_VERSION=12.4   version CUDA passee a nvfortran
    FORCE_REBUILD=1     ignore le cache des objets deja compiles
"""
from __future__ import annotations

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

if not PATHS.IS_WINDOWS:
    os.environ.setdefault("CC", "/usr/bin/gcc")
    os.environ.setdefault("CXX", "/usr/bin/g++")


# ============================================================
# Utilitaires
# ============================================================
def run(cmd, **kwargs):
    """Affiche puis execute une commande (leve une erreur si echec)."""
    cmd = [str(c) for c in cmd]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def require_file(path, label):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} source does not exist: {path}")


def is_stale(target: Path, *sources) -> bool:
    """True si `target` doit etre reconstruit (absent ou plus ancien qu'une source)."""
    if PATHS.FORCE_REBUILD or not target.exists():
        return True
    mtime = target.stat().st_mtime
    return any(Path(s).stat().st_mtime > mtime for s in sources)


def append_unique(lst: list, value):
    if value not in lst:
        lst.append(value)


def relative_sources(sources):
    """setuptools exige des chemins relatifs (POSIX) pour `sources`."""
    return [os.path.relpath(s, start=HERE).replace("\\", "/") for s in sources]


for _src in PATHS.CPP_SOURCES:
    require_file(_src, "C++")


# ============================================================
# Fortran CPU (gfortran)
# ============================================================
def compile_fortran_cpu(sources, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    compiler = "gfortran" if PATHS.IS_WINDOWS else str(PATHS.GFORTRAN)
    objects = []

    for src in map(Path, sources):
        require_file(src, "Fortran")
        obj = out_dir / f"{src.stem}{PATHS.OBJ_SUFFIX}"
        objects.append(str(obj))

        if not is_stale(obj, src):
            print(f"Skipping unchanged Fortran source: {src}")
            continue

        flags = ["-O3", "-march=native"]
        if PATHS.IS_WINDOWS:
            flags.append("-ffast-math")
        else:
            flags += ["-funroll-loops", "-fPIC", "-fopenmp"]

        if src.suffix.lower() in (".f", ".for"):  # format fixe (legacy)
            flags.append("-ffixed-line-length-none")
            if not PATHS.IS_WINDOWS:
                flags.append("-std=legacy")
        elif not PATHS.IS_WINDOWS:
            flags.append()

        run([compiler, "-c", src, "-o", obj, *flags])

    return objects


# ============================================================
# Fortran GPU (nvfortran + OpenACC) -- Linux uniquement
# ============================================================
def build_fortran_gpu_shared(sources, out_dir: Path) -> Path:
    """
    Compile + lie les sources OpenACC en bibliotheque partagee.
    nvfortran fait lui-meme le device-link (nvlink) et enregistre
    le binaire GPU aupres du runtime OpenACC.
    """
    if not PATHS.NVFORTRAN.is_file():
        raise RuntimeError(f"nvfortran not found: {PATHS.NVFORTRAN}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for s in sources:
        require_file(s, "Fortran")

    lib = out_dir / PATHS.GPU_LIB_FILE
    if not is_stale(lib, *sources):
        print(f"Skipping unchanged GPU library: {lib}")
        return lib

    run([
        PATHS.NVFORTRAN, "-O3", "-fPIC", "-shared", "-acc=gpu",
        f"-gpu=cuda{PATHS.CUDA_VERSION},cc{PATHS.CUDA_ARCH}",
        *map(str, sources),
        "-o", lib,
        f"-Wl,-rpath,{PATHS.NVHPC_LIB}",
        f"-Wl,-rpath,{PATHS.CUDA_LIB}",
    ])
    return lib


# ============================================================
# CUDA (nvcc)
# ============================================================
def compile_cuda(sources, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Les objets sont nommes d'apres le stem : on refuse les collisions.
    stems = [Path(s).stem for s in sources]
    if len(stems) != len(set(stems)):
        raise ValueError(f"CUDA sources with duplicate file names: {stems}")

    include_dirs = [
        PATHS.CUDA_INCLUDE,
        getattr(PATHS, "EVE_INCLUDE", None),
        PATHS.KOKKOS_INCLUDE,
        pybind11.get_include(),
        sysconfig.get_path("include"),
    ]
    include_args = [arg for d in include_dirs if d for arg in ("-I", str(d))]

    flags = [
        "-std=c++20",
        "--extended-lambda",
        "-O3",
        f"--generate-code=arch=compute_{PATHS.CUDA_ARCH},code=[sm_{PATHS.CUDA_ARCH},compute_{PATHS.CUDA_ARCH}]",
    ]
    env = os.environ.copy()

    if PATHS.IS_WINDOWS:
        msvc_bin = str(PATHS.MSVC_BIN)
        if os.path.exists(msvc_bin):
            env["PATH"] = msvc_bin + os.pathsep + env["PATH"]
        else:
            print(f"WARNING: MSVC path not found: {msvc_bin} (PATH systeme utilise)")
        flags += [
            "-ccbin", msvc_bin,
            "--machine", "64",
            "-cudart", "static",
            "-allow-unsupported-compiler",
            "-Wext-lambda-captures-this",
            "--use_fast_math",
            "-Xcompiler", "/EHsc,/O2,/MT,/Zc:preprocessor,/bigobj,/permissive,/openmp",
        ]
    else:
        flags += ["-Xcompiler", "-fPIC"]

    objects = []
    for src in sources:
        src = Path(src)
        if not src.is_absolute():
            src = (PATHS.MODULE_ROOT / src).resolve()
        require_file(src, "CUDA")

        obj = out_dir / f"{src.stem}{PATHS.OBJ_SUFFIX}"
        objects.append(str(obj))

        if not is_stale(obj, src):
            print(f"Skipping unchanged CUDA source: {src}")
            continue

        run([PATHS.NVCC, "-c", src, "-o", obj, *flags, *include_args], env=env)

    return objects


# ============================================================
# Configuration specifique a la plateforme
# ============================================================
LIBRARIES = ["kokkoscore", "cudart_static", "cuda", "cublas", "openblas", "lapack"]
# INSTALL_REQUIRES = ["numpy", "cupy-cuda12x", "numba", "pycuda", "psutil"]
INSTALL_REQUIRES = []
RUNTIME_LIBRARY_DIRS = []
EXTRA_LINK_ARGS = []
if PATHS.IS_WINDOWS:
    EXTRA_COMPILE_ARGS = [
        "/O2", "/EHsc", "/bigobj", "/arch:AVX2", "/MT",
        "/DEIGEN_USE_BLAS",
        "/DARMA_DONT_USE_WRAPPER", "/DARMA_DONT_USE_LAPACK", "/DARMA_USE_BLAS",
    ]
else:
    EXTRA_COMPILE_ARGS = [
        "-O3", "-fPIC", "-mavx2", "-fopenmp",
        "-DEIGEN_USE_BLAS",
        "-DARMA_USE_BLAS",
        "-DHAS_GPU_FORTRAN",  # FORTRAN/fortran.h
    ]
    EXTRA_LINK_ARGS += ["-Wl,--no-as-needed", "-lgomp"]
    LIBRARIES += [
        "gfortran", "quadmath", "dl", "pthread", "m",
    ]

# ============================================================
# build_ext personnalise
# ============================================================
class BuildCUDA(build_ext):

    def build_extensions(self):
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        objects = compile_fortran_cpu(PATHS.FORTRAN_CPU, build_temp / "fortran_cpu")
        objects += compile_cuda(PATHS.CUDA_SOURCES, build_temp / "cuda")

        gpu_lib = None
        if not PATHS.IS_WINDOWS:
            gpu_lib = build_fortran_gpu_shared(PATHS.FORTRAN_GPU, build_temp / "fortran_gpu")

        for ext in self.extensions:
            for obj in objects:
                append_unique(ext.extra_objects, obj)
            append_unique(ext.include_dirs, str(PATHS.CUDA_INCLUDE))
            append_unique(ext.library_dirs, str(PATHS.CUDA_LIB))

            if gpu_lib is not None:
                append_unique(ext.library_dirs, str(gpu_lib.parent))
                append_unique(ext.libraries, PATHS.GPU_LIB_NAME)
                # $ORIGIN : lib copiee a cote du .so ; MODULE_ROOT : builds en place / pip -e
                for rp in ("$ORIGIN", str(PATHS.MODULE_ROOT)):
                    append_unique(ext.runtime_library_dirs, rp)

        super().build_extensions()

        if gpu_lib is not None:
            self._copy_gpu_lib(gpu_lib)

    def _copy_gpu_lib(self, gpu_lib: Path):
        """Copie la lib GPU a cote de l'extension et dans MODULE_ROOT (survit a pip -e)."""
        for ext in self.extensions:
            dest_dirs = {
                Path(self.get_ext_fullpath(ext.name)).resolve().parent,
                Path(PATHS.MODULE_ROOT).resolve(),
            }
            for dest_dir in dest_dirs:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / PATHS.GPU_LIB_FILE
                print(f"Copying {gpu_lib} -> {dest}", flush=True)
                shutil.copy2(gpu_lib, dest)


# ============================================================
# Extension + package
# ============================================================
ext_modules = [
    Pybind11Extension(
        "hpc_benchmark_core",
        relative_sources(PATHS.CPP_SOURCES),
        include_dirs=[str(p) for p in PATHS.INCLUDE_DIRS],
        library_dirs=[str(p) for p in PATHS.LIBRARY_DIRS],
        libraries=LIBRARIES,
        cxx_std=20,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        runtime_library_dirs=RUNTIME_LIBRARY_DIRS,
    ),
]

setup(
    name="hpc_test_extensions",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildCUDA},
    install_requires=INSTALL_REQUIRES,
)
