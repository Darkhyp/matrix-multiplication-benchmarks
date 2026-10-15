 ! Copyright (c) 2026
 ! Author: Alexandre Korovin
 ! e-mail: a.v.korovin73@gmail.com
 ! SPDX-License-Identifier: MIT

subroutine matmul_cpu(A, B, C, N, M, P, elapsed_ms)
    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments passed from Python/C layout
    integer(c_int), intent(in) :: N, M, P
    real(c_float), intent(in)  :: A(N, M), B(M, P)
    real(c_float), intent(out) :: C(N, P)
    real(c_double), intent(out):: elapsed_ms

!    integer :: i, j, k
    integer :: i, j, l, jj, ll
    integer, parameter  :: BJ = 64, BL = 256

    integer(c_long_long) :: start_count, end_count, rate

    ! Start CPU Timer
    call system_clock(start_count, rate)

    c = 0.0_c_float
    !$omp parallel do private(ll, j, l, i) schedule(static)
    do jj = 1, p, BJ
        do ll = 1, m, BL
          do j = jj, min(jj+BJ-1, p)
            do l = ll, min(ll+BL-1, m)
              !$omp simd
              do i = 1, n
                c(i,j) = c(i,j) + a(i,l) * b(l,j)
              end do
            end do
          end do
        end do
    end do

    ! End Timer
    call system_clock(end_count)

    elapsed_ms = (real(end_count - start_count, kind=c_double) / real(rate, kind=c_double)) * 1000.0_c_double

end subroutine matmul_cpu