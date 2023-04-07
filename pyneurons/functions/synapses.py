from jax.random import normal


def synapses(key, shape):
    return normal(key, shape) * 0.1 + 1.5
