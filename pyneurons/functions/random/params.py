from jax.random import normal


def params(key, shape):
    return normal(key, shape) * 0.1 + 1.5
