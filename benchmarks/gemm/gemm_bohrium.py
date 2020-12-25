import bohrium as bh


def gemm(a, b):
    return bh.linalg.matmul(a, b)


def prepare_inputs(a, b, device):
    out = [bh.array(k) for k in (a, b)]
    for o in out:
        # force allocation on target device
        tmp = o * 1  # noqa: F841
        bh.flush()
    return out


def run(a, b, device='cpu'):
    out = gemm(a, b)
    bh.flush()
    return out
