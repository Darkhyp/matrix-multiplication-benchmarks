#nvfortran -O2 -acc=gpu -gpu=cuda12.4,cc89 \
#    tests/test_gpu.f90 \
#    -o tests/test_gpu
#
#./tests/test_gpu
#
#nvfortran -O2 -acc=gpu -gpu=cuda12.4,cc89 \
#    -c /workspace/test_module/src/FORTRAN/matrix_dot2.f90 \
#    -o tests/matrix_dot2.o
#
#nvfortran -O2 -acc=gpu -gpu=cuda12.4,cc89 \
#    tests/test_matmul.f90 tests/matrix_dot2.o \
#    -o tests/test_matmul
#
#./tests/test_matmul
#
#/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/bin/nvfortran \
#    -O2 -acc=gpu -gpu=cuda12.4,cc89 -Minfo=accel \
#    -c /workspace/test_module/src/FORTRAN/matrix_dot2.f90 \
#    -o /workspace/tests/matrix_dot2.o
#
#/workspace/tests/test_matmul
#echo "Code de sortie : $?"


nvfortran -acc=gpu -gpu=cuda12.4,cc89 -dryrun \
    /workspace/tests/test_matmul.f90 \
    /workspace/tests/matrix_dot2.o \
    -o /workspace/tests/test_matmul_dryrun \
    2>&1 | tee /workspace/nvfortran_link.log

grep -Ei 'acc|nvf|nvomp|nvhpc|cuda|link' \
    /workspace/nvfortran_link.log

ldd /workspace/test_module_linux/hpc_benchmark_core.cpython-310-x86_64-linux-gnu.so \
    | grep -Ei 'acc|nvf|nvomp|nvhpc|cuda'