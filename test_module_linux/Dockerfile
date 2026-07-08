# syntax=docker/dockerfile:1.7

FROM nvcr.io/nvidia/nvhpc:24.5-devel-cuda12.4-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# NVIDIA HPC SDK
# -----------------------------------------------------------------------------
ENV NVARCH=Linux_x86_64
ENV HPC_VERSION=24.5
ENV CUDA_VERSION=12.4

ENV NVCOMPILER=/opt/nvidia/hpc_sdk/${NVARCH}/${HPC_VERSION}

ENV CUDA_HOME=${NVCOMPILER}/cuda/${CUDA_VERSION}
ENV CUDA_ROOT=${CUDA_HOME}
ENV CUDA_PATH=${CUDA_HOME}
ENV CUDAToolkit_ROOT=${CUDA_HOME}

ENV MATH_LIB_DIR=${NVCOMPILER}/math_libs/${CUDA_VERSION}/targets/x86_64-linux/lib


# -----------------------------------------------------------------------------
# CUDA / HPC environment
# -----------------------------------------------------------------------------
ENV CUDACXX=${CUDA_HOME}/bin/nvcc

ENV PATH=/usr/lib/ccache:/usr/bin:${NVCOMPILER}/compilers/bin:${CUDA_HOME}/bin:${PATH}

ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MATH_LIB_DIR}:${NVCOMPILER}/compilers/lib:${LD_LIBRARY_PATH}

ENV LIBRARY_PATH=${MATH_LIB_DIR}:${LIBRARY_PATH}

ENV LDFLAGS=-L${MATH_LIB_DIR}

ENV CPATH=${CUDA_HOME}/include:${CPATH}


# -----------------------------------------------------------------------------
# Use GCC/G++ for Python/C++ extensions
# (PyCUDA, pybind11, numpy extensions, etc.)
# -----------------------------------------------------------------------------
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++


# -----------------------------------------------------------------------------
# CUDA compatibility path
# -----------------------------------------------------------------------------
RUN ln -sfn ${CUDA_HOME} /usr/local/cuda


# -----------------------------------------------------------------------------
# System packages
# -----------------------------------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        wget \
        ca-certificates \
        gpg \
        build-essential \
        ccache \
        ninja-build \
        git \
        python3-dev \
        python3-pip \
        gcc-12 \
        g++-12 \
        gfortran-12 \
        libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*


# -----------------------------------------------------------------------------
# GCC 12 as default compiler
# -----------------------------------------------------------------------------
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc \
        x86_64-linux-gnu-gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ \
        x86_64-linux-gnu-g++ /usr/bin/g++-12 100 && \
    update-alternatives --install /usr/bin/gfortran \
        gfortran /usr/bin/gfortran-12 100


# -----------------------------------------------------------------------------
# Latest CMake
# -----------------------------------------------------------------------------
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
        2>/dev/null | \
        gpg --dearmor \
        -o /usr/share/keyrings/kitware-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] \
        https://apt.kitware.com/ubuntu/ jammy main" \
        > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends cmake && \
    rm -rf /var/lib/apt/lists/*


# -----------------------------------------------------------------------------
# Python tools
# -----------------------------------------------------------------------------
RUN python3 -m pip install \
    --root-user-action=ignore \
    --disable-pip-version-check \
    --no-cache-dir \
    --upgrade \
    pip \
    setuptools \
    wheel

RUN python3 -m pip install \
    numpy \
    pybind11 \
    numba \
    cupy-cuda12x


# -----------------------------------------------------------------------------
# Building PyCUDA \
# -----------------------------------------------------------------------------
RUN git clone https://github.com/inducer/pycuda.git /tmp/pycuda && \
    cd /tmp/pycuda && \
    python3 configure.py --cuda-root=${CUDA_HOME} && \
    sed -i \
        "s|CUDA_ENABLE_CURAND = True|CUDA_ENABLE_CURAND = True|" \
        siteconf.py && \
    python3 -m pip install . --no-build-isolation && \
    rm -rf /tmp/pycuda


# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
RUN python3 - <<'EOF'
import pycuda
print("PyCUDA version:", pycuda.VERSION_TEXT)
print("PyCUDA installation OK")
EOF


# -----------------------------------------------------------------------------
# Workspace
# -----------------------------------------------------------------------------
WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]
#CMD ["bash"]