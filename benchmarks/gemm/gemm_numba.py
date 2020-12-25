import numpy as np
import numba as nb


@nb.jit(nopython=True, fastmath=True)
def gemm(out, a, b):
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                out[i][j] += a[i][k] * b[k][j]


def run(a, b, device='cpu'):
    out = np.empty_like(a)
    gemm(out, a, b)
    return out
