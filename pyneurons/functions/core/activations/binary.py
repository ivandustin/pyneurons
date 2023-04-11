from jax.numpy import where


def binary(x):
    return where(x >= 1.0, 1.0, 0.0)
