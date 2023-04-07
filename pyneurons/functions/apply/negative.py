from jax.numpy import negative as negative_function
from .positive import positive
from jax import jit


@jit
def negative(matrix, x):
    return negative_function(positive(matrix, x))