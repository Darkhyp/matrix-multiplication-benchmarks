#!/bin/bash
set -euo pipefail

HPC_VERSION="24.5"
HPC_ROOT="/opt/nvidia/hpc_sdk/Linux_x86_64/${HPC_VERSION}"
CUDA_DIR="${HPC_ROOT}/cuda/12.4"
MATH_DIR="${HPC_ROOT}/math_libs/12.4/targets/x86_64-linux"
PYCUDA_VERSION="2024.1.2"   # first line to drop the Boost.Python dependency (uses pybind11 instead)

echo "============================================================"
echo " HPC + CUDA + PyCUDA Compatible Installer (WSL)"
echo "============================================================"

###############################################################
# SYSTEM PACKAGES
###############################################################

sudo apt-get update

APT_PACKAGES=(
    wget curl gnupg2 build-essential g++ gfortran make
    libopenblas-dev python3-dev python3-pip python3-venv
)

MISSING=()
for pkg in "${APT_PACKAGES[@]}"; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Installing missing system packages: ${MISSING[*]}"
    sudo apt-get install -y "${MISSING[@]}"
else
    echo "All required system packages already present."
fi

###############################################################
# INSTALL HPC SDK
###############################################################

if [ ! -d "$HPC_ROOT" ]; then
    echo "Installing NVIDIA HPC SDK..."

    sudo mkdir -p /usr/share/keyrings

    if [ ! -f /usr/share/keyrings/nvidia-hpcsdk.gpg ]; then
        curl -fsSL https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk.gpg
    fi

    if [ ! -f /etc/apt/sources.list.d/nvhpc.list ]; then
        echo "deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /" \
        | sudo tee /etc/apt/sources.list.d/nvhpc.list >/dev/null
    fi

    sudo apt-get update
    sudo apt-get install -y nvhpc-24-5
else
    echo "HPC SDK already installed"
fi

###############################################################
# CREATE CUDA TOOLKIT VIEW (CRITICAL FIX FOR PYCUDA)
###############################################################

echo "Creating unified CUDA layout for PyCUDA..."

sudo mkdir -p "$CUDA_DIR/include"
sudo mkdir -p "$CUDA_DIR/lib64"

# Merge every math_libs include/lib tree we can find (layouts vary between
# NVHPC releases: some use math_libs/<ver>/include, others nest an extra
# targets/x86_64-linux/ directory).
for d in \
    "$HPC_ROOT/math_libs/12.4/include" \
    "$HPC_ROOT/math_libs/12.4/targets/x86_64-linux/include" \
    "$HPC_ROOT/math_libs/include"
do
    if [ -d "$d" ]; then
        echo "  merging headers from $d"
        sudo cp -rsf "$d/." "$CUDA_DIR/include/"
    fi
done

for d in \
    "$HPC_ROOT/math_libs/12.4/lib64" \
    "$HPC_ROOT/math_libs/12.4/targets/x86_64-linux/lib" \
    "$HPC_ROOT/math_libs/lib64"
do
    if [ -d "$d" ]; then
        echo "  merging libs from $d"
        sudo cp -rsf "$d/." "$CUDA_DIR/lib64/"
    fi
done

# Safety net: if curand.h (or other math headers pycuda needs) still isn't
# present, search the whole HPC SDK tree for it and copy it in directly.
for hdr in curand.h cublas.h cufft.h cusparse.h cusolver.h; do
    if [ ! -f "$CUDA_DIR/include/$hdr" ]; then
        found="$(sudo find "$HPC_ROOT" -maxdepth 8 -iname "$hdr" 2>/dev/null | head -n1)"
        if [ -n "$found" ]; then
            echo "  found $hdr at $found, copying in"
            sudo cp -f "$found" "$CUDA_DIR/include/$hdr"
        else
            echo "  WARNING: $hdr not found anywhere under $HPC_ROOT"
        fi
    fi
done

###############################################################
# /usr/local/cuda SYMLINK
###############################################################

if [ -L /usr/local/cuda ] || [ -e /usr/local/cuda ]; then
    sudo rm -rf /usr/local/cuda
fi
sudo ln -s "$CUDA_DIR" /usr/local/cuda

###############################################################
# ENVIRONMENT VARIABLES
###############################################################

BASHRC_MARKER="### HPC_CUDA_ENV ###"

if ! grep -q "$BASHRC_MARKER" ~/.bashrc; then
cat >> ~/.bashrc <<EOF

$BASHRC_MARKER
export NVHPC=$HPC_ROOT
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda

export PATH=\$NVHPC/compilers/bin:\$CUDA_HOME/bin:\$PATH
export CPATH=\$CUDA_HOME/include:\$CPATH
export CPLUS_INCLUDE_PATH=\$CUDA_HOME/include:\$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=\$CUDA_HOME/lib64:\$LIBRARY_PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
EOF
fi

export NVHPC="$HPC_ROOT"
export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda"
export PATH="$NVHPC/compilers/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

###############################################################
# SANITY CHECK: nvcc + headers must exist before we try PyCUDA
###############################################################

if [ ! -x "$CUDA_DIR/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at $CUDA_DIR/bin/nvcc"
    echo "The HPC SDK install did not produce the expected CUDA toolkit layout."
    exit 1
fi

if [ ! -f "$CUDA_DIR/include/cuda.h" ]; then
    echo "ERROR: cuda.h not found at $CUDA_DIR/include/cuda.h"
    exit 1
fi

###############################################################
# PYTHON SETUP
###############################################################

if [ -n "${VIRTUAL_ENV:-}" ]; then
    PIP=pip
    PY=python
else
    PIP=pip3
    PY=python3
fi

$PIP install --upgrade pip setuptools wheel

###############################################################
# CHECK + INSTALL PYTHON BUILD DEPENDENCIES
# (required because --no-build-isolation skips pip's automatic
#  build dependency resolution, so they must already be present)
###############################################################

PY_BUILD_DEPS=(numpy mako pytools pybind11 appdirs platformdirs decorator)

echo "Checking Python build dependencies for PyCUDA..."
for dep in "${PY_BUILD_DEPS[@]}"; do
    if $PY -c "import ${dep}" >/dev/null 2>&1; then
        echo "  $dep: OK"
    else
        echo "  $dep: missing -> installing"
        $PIP install --no-cache-dir "$dep"
    fi
done

###############################################################
# INSTALL / VERIFY PYCUDA
###############################################################

CURRENT_PYCUDA_VERSION="$($PY -c 'import pycuda; print(pycuda.VERSION_TEXT)' 2>/dev/null || true)"

if [ "$CURRENT_PYCUDA_VERSION" = "$PYCUDA_VERSION" ]; then
    echo "PyCUDA $PYCUDA_VERSION already installed and importable, skipping rebuild."
else
    if [ -n "$CURRENT_PYCUDA_VERSION" ]; then
        echo "Found PyCUDA $CURRENT_PYCUDA_VERSION, replacing with $PYCUDA_VERSION..."
        $PIP uninstall -y pycuda || true
    fi

    echo "Building PyCUDA ${PYCUDA_VERSION} against CUDA at $CUDA_HOME ..."
    $PIP install --no-cache-dir --no-build-isolation "pycuda==${PYCUDA_VERSION}"
fi

# cupy is independent of pycuda's build; install if missing
if ! $PY -c "import cupy" >/dev/null 2>&1; then
    echo "Installing cupy-cuda12x..."
    $PIP install cupy-cuda12x
else
    echo "cupy: OK"
fi

###############################################################
# VERIFY INSTALLATION
###############################################################

echo
echo "============================================================"
echo " VERIFY"
echo "============================================================"

echo "CUDA headers:"
[ -f "$CUDA_HOME/include/cuda.h" ] && echo "  cuda.h OK" || echo "  cuda.h MISSING"
[ -f "$CUDA_HOME/include/curand.h" ] && echo "  curand.h OK" || echo "  curand.h MISSING"

echo
echo "nvcc:"
nvcc --version | head -2

echo
echo "Python:"
$PY --version

echo
echo "G++:"
g++ --version | head -1

echo
echo "PyCUDA import test:"
$PY -c "import pycuda.driver as cuda; cuda.init(); print('PyCUDA OK, version', __import__('pycuda').VERSION_TEXT); print('Devices found:', cuda.Device.count())"

echo
echo "DONE"
echo "Restart terminal or run: source ~/.bashrc"