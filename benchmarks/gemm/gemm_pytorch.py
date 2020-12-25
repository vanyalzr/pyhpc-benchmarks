import numpy as np
import torch


@torch.jit.script
def gemm(a, b):
    return torch.mm(a, b)


def prepare_inputs(a, b, device):
    out = [torch.as_tensor(x, device='cuda'
                              if device == 'gpu' else 'cpu') for x in (a, b)]
    if device == 'gpu':
        torch.cuda.synchronize()
    return out


def run(a, b, device='cpu'):
    with torch.no_grad():
        out = gemm(a, b)
    if device == 'gpu':
        torch.cuda.synchronize()
    return out
