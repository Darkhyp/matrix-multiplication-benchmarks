 ! Copyright (c) 2026
 ! Author: Alexandre Korovin
 ! e-mail: a.v.korovin73@gmail.com
 ! SPDX-License-Identifier: MIT

subroutine matmul_cpu_sgemm(A, B, C, N, M, P, elapsed_ms)
    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments passed from Python/C layout
    integer(c_int), intent(in) :: N, M, P
    real(c_float), intent(in)  :: A(N, M), B(M, P)
    real(c_float), intent(out) :: C(N, P)
    real(c_double), intent(out):: elapsed_ms

!    integer :: i, j, k
    integer :: i, j, l, jj, ll
    integer(c_long_long) :: start_count, end_count, rate

    ! Start CPU Timer
    call system_clock(start_count, rate)

    call sgemm('N', 'N', N, P, M, 1.0_c_float, A, N, B, M, 0.0_c_float, C, N)

    ! End Timer
    call system_clock(end_count)

    elapsed_ms = (real(end_count - start_count, kind=c_double) / real(rate, kind=c_double)) * 1000.0_c_double

end subroutine matmul_cpu_sgemm