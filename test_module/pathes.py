import os

MSVC_BIN_DIR = r"C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Tools\MSVC\14.50.35717\bin\HostX86\x64"

CUDA_PATH = os.environ.get(
    "CUDA_PATH",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
)
CUDA_INCLUDE = os.path.join(CUDA_PATH, "include")
CUDA_LIB_DIR = os.path.join(CUDA_PATH, "lib", "x64")
NVCC = os.path.join(CUDA_PATH, "bin", "nvcc.exe")

# Dynamically find the path relative to your setup.py script
current_dir = os.path.dirname(os.path.abspath(__file__))
EVE_INCLUDE = os.path.join(current_dir, "EVE_SRC", "include")

KIWAKU_INCLUDE = os.path.join(current_dir, "KIWAKU_SRC", "include")
RABERU_INCLUDE = os.path.join(current_dir, "RABERU_SRC", "include")
KUMI_INCLUDE = os.path.join(current_dir, "KUMI_SRC", "include")

KOKKOS_DIR = os.path.join(current_dir, "KOKKOS_SRC", "dist")
KOKKOS_INCLUDE = os.path.join(KOKKOS_DIR, "include")
KOKKOS_LIB_DIR = os.path.join(KOKKOS_DIR, "lib")

MSYS_LIB_DIR = r"C:\msys64\ucrt64\lib"