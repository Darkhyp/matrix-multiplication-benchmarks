program test

use iso_c_binding

integer, parameter :: N=16, M=16, P=16

real(c_float) :: A(N,M), B(M,P), C(N,P)
real(c_double) :: t

A = 1
B = 2

call matmul_gpu(A, B, C, N, M, P, t)

print *, C(1,1)

end program