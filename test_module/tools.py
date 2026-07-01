import os
import platform
import cupy as cp
import psutil


def print_GPU_info():
    if cp.cuda.is_available():
        device_count = cp.cuda.runtime.getDeviceCount()
        props = cp.cuda.runtime.getDeviceProperties(0)
        # Decode byte string name if necessary (depends on CuPy version environment)
        gpu_name = props["name"].decode('utf-8') if isinstance(props["name"], bytes) else props["name"]

        print(f"GPU: CUDA is available!")
        print(f"\tDevice Count: {device_count}")
        print(f"\tPrimary Device: {gpu_name}")
        print(f"\tCompute Capability: {props['major']}.{props['minor']}")
        print(f"\tTotal VRAM: {props['totalGlobalMem'] / (1024 ** 3):.2f} GB")
    else:
        print("GPU: CUDA is not available! (Falling back to host execution layout)")


def print_system_info():
    print("--- System Environment info ---")
    # Gather System & OS Info
    print(f"OS: {platform.system()} {platform.release()} (v{platform.version()})")
    print(f"Architecture: {platform.machine()}")

    # Gather CPU Info
    print(f"CPU: {platform.processor()}")
    print(f"\tPhysical Cores: {psutil.cpu_count(logical=False)}")
    print(f"\tLogical Threads: {psutil.cpu_count(logical=True)}")
    print(f"\tMax Frequency: {psutil.cpu_freq().max:.2f} MHz" if psutil.cpu_freq() else "\tMax Frequency: N/A")

    # Gather System Memory (RAM)
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"System RAM: {total_ram_gb:.2f} GB")

    print("\n--- Compute Device Info ---")
    # Gather Accelerator Info (Your original logic updated)
    print_GPU_info()
    print()
