import numpy as np
import tensorflow as tf


def gemm(a, b):
    return tf.linalg.matmul(a, b)


gemm_tf = tf.function(gemm, experimental_compile=True)


def prepare_inputs(a, b, device):
    return [tf.convert_to_tensor(x) for x in (a, b)]


def run(a, b, device='cpu'):
    out = gemm_tf(a, b)
    return out
