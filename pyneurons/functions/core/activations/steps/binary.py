from jax.numpy import heaviside


def binary(x):
    return heaviside(x, 1)
