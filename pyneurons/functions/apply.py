from jax import jit
from .activations import spike as f


@jit
def apply(m, b, a, x):
    return f(((x @ m) * a) + b)
