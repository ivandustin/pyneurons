from jax.numpy import where, minimum


def spike(x):
    return where(x >= 1, minimum(x, 2), 0)
