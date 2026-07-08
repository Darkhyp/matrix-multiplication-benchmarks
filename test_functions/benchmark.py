# Copyright (c) 2026
# Author: Alexandre Korovin
# e-mail: a.v.korovin73@gmail.com
# SPDX-License-Identifier: MIT

import functools
import math

def benchmark(text, n=10, warmup=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(A, B):
            if warmup:
                func(A, B)

            times = []
            result = None

            for _ in range(n):
                execution_time, result = func(A, B)
                times.append(execution_time)

            mean = sum(times) / len(times)
            rms = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))

            return {
                "comment": text,
                "mean": mean,
                "rms": rms,
                "runs": times,
                "result": result,
            }

        return wrapper

    return decorator