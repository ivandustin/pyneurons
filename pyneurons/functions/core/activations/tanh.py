from jax.numpy import clip


def tanh(x):
    return clip(x, -1, 1)
