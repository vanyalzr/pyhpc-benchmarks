import jax
import jax.numpy as jnp


@jax.jit
def gemm(a, b):
    return jnp.dot(a, b)


def prepare_inputs(a, b, device):
    out = [jnp.array(k) for k in (a, b)]
    for o in out:
        o.block_until_ready()
    return out


def run(a, b, device='cpu'):
    out = gemm(a, b)
    out.block_until_ready()
    return out
