import math
import importlib
import functools


def generate_inputs(size):
    import numpy as np
    np.random.seed(24)

    a = np.random.uniform(-1, 1, size=(size, size))
    b = np.random.uniform(-1, 1, size=(size, size))
    return a, b


def try_import(backend):
    try:
        return importlib.import_module(f'.gemm_{backend}', __name__)
    except ImportError:
        return None


def get_callable(backend, size, device='cpu'):
    backend_module = try_import(backend)
    inputs = generate_inputs(size)
    if hasattr(backend_module, 'prepare_inputs'):
        inputs = backend_module.prepare_inputs(*inputs, device=device)
    return functools.partial(backend_module.run, *inputs, device=device)


__implementations__ = (
    'numpy',
    'pytorch',
    'jax',
    'tensorflow',
    'numba',
    'bohrium',
)
