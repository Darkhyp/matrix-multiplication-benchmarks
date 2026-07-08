# Copyright (c) 2026
# Author: Alexandre Korovin
# e-mail: a.v.korovin73@gmail.com
# SPDX-License-Identifier: MIT

try:
    from .test_module import eve_dot, eve_dot2, cuda_dot, kokkos_dot, kiwaku_dot
except ImportError:
    print("Cannot import test_module function for windows")