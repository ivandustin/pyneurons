from jax.numpy import clip


def sigmoid(x):
    return clip(x, 0, 1)
