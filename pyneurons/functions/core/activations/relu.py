from jax.numpy import maximum


def relu(x):
    return maximum(x, 0)
