subroutine matmul_cpu(A, B, C, N, M, P, elapsed_ms)
    use, intrinsic :: iso_c_binding
    implicit none

    ! Arguments passed from Python/C layout
    integer(c_int), intent(in) :: N, M, P
    real(c_float), intent(in)  :: A(N, M), B(M, P)
    real(c_float), intent(out) :: C(N, P)
    real(c_double), intent(out):: elapsed_ms

    integer :: i, j, k
    integer(c_long_long) :: start_count, end_count, rate

    ! Start CPU Timer
    call system_clock(start_count, rate)

    ! Modern Standard Parallel Loop (or use OpenMP: !$omp parallel do collapse(2))
    do concurrent (j = 1:P, i = 1:N)
        C(i, j) = 0.0_c_float  ! FIX: Changed '0.0pt' to correct Fortran kind literal syntax
        do k = 1, M
            C(i, j) = C(i, j) + A(i, k) * B(k, j)
        end do
    end do

    ! End Timer
    call system_clock(end_count)

    elapsed_ms = (real(end_count - start_count, kind=c_double) / real(rate, kind=c_double)) * 1000.0_c_double

end subroutine matmul_cpu