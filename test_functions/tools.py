# Copyright (c) 2026
# Author: Alexandre Korovin
# e-mail: a.v.korovin73@gmail.com
# SPDX-License-Identifier: MIT

import platform
import sys

import cupy as cp
import psutil


def _get_cpu_name():
    """Return a human-readable CPU name on Windows and Linux."""
    cpu = platform.processor()

    if cpu:
        return cpu

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass

    return "Unknown"


def _cuda_version(version):
    """Convert CUDA integer version (e.g. 12040) -> '12.4'."""
    major = version // 1000
    minor = (version % 1000) // 10
    return f"{major}.{minor}"


def print_gpu_info():
    """Print information about all CUDA devices."""

    try:
        if not cp.cuda.is_available():
            print("GPU : CUDA is not available.")
            return

        device_count = cp.cuda.runtime.getDeviceCount()

        print(f"CUDA Devices : {device_count}")

        try:
            driver = cp.cuda.runtime.driverGetVersion()
            runtime = cp.cuda.runtime.runtimeGetVersion()

            print(f"CUDA Driver  : {_cuda_version(driver)}")
            print(f"CUDA Runtime : {_cuda_version(runtime)}")
        except Exception:
            pass

        print(f"CuPy Version : {cp.__version__}")

        try:
            import pycuda
            print(f"PyCUDA       : {pycuda.VERSION_TEXT}")
        except Exception:
            pass

        print()

        for i in range(device_count):
            props = cp.cuda.runtime.getDeviceProperties(i)

            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode()

            print(f"GPU {i}")
            print(f"    Name               : {name}")
            print(
                f"    Compute Capability : {props['major']}.{props['minor']}"
            )
            print(
                f"    Total VRAM         : {props['totalGlobalMem'] / (1024 ** 3):.2f} GB"
            )

    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"CUDA Runtime Error : {e}")


def print_system_info():
    """Print host system information."""

    print("=" * 70)
    print("System Information")
    print("=" * 70)

    print(f"OS              : {platform.system()} {platform.release()}")
    print(f"Architecture    : {platform.machine()}")
    print(f"Python          : {sys.version.split()[0]}")
    print(f"CPU             : {_get_cpu_name()}")

    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)

    print(f"Physical Cores  : {physical}")
    print(f"Logical Threads : {logical}")

    freq = psutil.cpu_freq()
    if freq is not None:
        print(f"CPU Frequency   : {freq.current:.2f} MHz")

    ram = psutil.virtual_memory().total / (1024 ** 3)
    print(f"System RAM      : {ram:.2f} GB")

    print()
    print("=" * 70)
    print("CUDA Information")
    print("=" * 70)

    print_gpu_info()

    print("=" * 70)