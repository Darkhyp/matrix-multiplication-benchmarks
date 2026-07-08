try:
    from .test_module import eve_dot, eve_dot2, cuda_dot, kokkos_dot, kiwaku_dot
except ImportError:
    print("Cannot import test_module function for windows")