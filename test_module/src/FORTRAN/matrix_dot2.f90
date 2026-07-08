 ! Copyright (c) 2026
 ! Author: Alexandre Korovin
 ! e-mail: a.v.korovin73@gmail.com
 ! SPDX-License-Identifier: MIT

 subroutine matmul_gpu(A, B, C, N, M, P, elapsed_ms)
    use, intrinsic :: iso_c_binding
    implicit none

    integer(c_int), intent(in) :: N, M, P
    real(c_float), intent(in)  :: A(N, M), B(M, P)
    real(c_float), intent(out) :: C(N, P)
    real(c_double), intent(out):: elapsed_ms

    integer :: i, j, k
    integer(c_long_long) :: start_count, end_count, rate

    ! Start CPU/GPU sync tracking timer
    call system_clock(start_count, rate)

    ! nvfortran automatically converts this block into a CUDA Kernel Grid
    ! shared memory management and thread configurations are handled by the compiler
!    do concurrent (j = 1:P, i = 1:N) local(k)
!        C(i, j) = 0.0_c_float
!        do k = 1, M
!            C(i, j) = C(i, j) + A(i, k) * B(k, j)
!        end do
!    end do
    do j = 1, P
        do i = 1, N
            C(i, j) = 0.0_c_float
            do k = 1, M
                C(i, j) = C(i, j) + A(i, k) * B(k, j)
            end do
        end do
    end do

    call system_clock(end_count)
    elapsed_ms = (real(end_count - start_count, c_double) / real(rate, c_double)) * 1000.0_c_double
end subroutine matmul_gpu