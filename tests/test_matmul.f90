program test_matmul
    use iso_c_binding
    implicit none

    integer(c_int), parameter :: N = 256, M = 256, P = 256
    real(c_float), allocatable :: A(:,:), B(:,:), C(:,:)
    real(c_double) :: elapsed_ms
    integer :: i, j
    real(c_float) :: err

    allocate(A(N,M), B(M,P), C(N,P))

    A = 1.0_c_float
    B = 2.0_c_float

    call matmul_gpu(A, B, C, N, M, P, elapsed_ms)

    err = maxval(abs(C - real(M * 2, c_float)))

    print *, "C(1,1) =", C(1,1)
    print *, "C(N,P) =", C(N,P)
    print *, "Max error =", err
    print *, "Elapsed (ms) =", elapsed_ms

    if (err > 1.e-4_c_float) error stop "Incorrect result"

    print *, "Fortran matrix multiplication: OK"
end program test_matmul
