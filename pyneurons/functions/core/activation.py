from jax.numpy import where, minimum


def activation(x):
    return where(x >= 1, minimum(x, 2), 0)
