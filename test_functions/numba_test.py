# Copyright (c) 2026
# Author: Alexandre Korovin
# e-mail: a.v.korovin73@gmail.com
# SPDX-License-Identifier: MIT

import numpy as np
from numba import njit, prange


@njit(fastmath=True, cache=True)
def numba_matrix_dot(A, B):
    rows_A, cols_A, cols_B = A.shape[0], A.shape[1], B.shape[1]

    # Initialize a 2D output array instead of a flat 1D array
    C = np.zeros((rows_A, cols_B), dtype=np.float32)

    # Standard, clean IKJ loop order
    for i in range(rows_A):
        for k in range(cols_A):
            a_ik = A[i, k]  # Clean 2D scalar access

            for j in range(cols_B):
                # Clean 2D addition: scalar += scalar * scalar
                C[i, j] += a_ik * B[k, j]

    return C