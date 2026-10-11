program test_gpu
    use iso_fortran_env, only: real32
    implicit none

    integer, parameter :: n = 1024
    real(real32) :: a(n), b(n), c(n)
    integer :: i

    do i = 1, n
        a(i) = real(i, real32)
        b(i) = 2.0_real32
    end do

    !$acc parallel loop copyin(a, b) copyout(c)
    do i = 1, n
        c(i) = a(i) * b(i)
    end do
    !$acc wait

    print *, "c(1)   =", c(1)
    print *, "c(1024)=", c(n)

    if (abs(c(1) - 2.0_real32) > 1.e-5_real32 .or. &
        abs(c(n) - 2048.0_real32) > 1.e-5_real32) then
        error stop "GPU computation failed"
    end if

    print *, "Fortran OpenACC GPU test: OK"
end program test_gpu
