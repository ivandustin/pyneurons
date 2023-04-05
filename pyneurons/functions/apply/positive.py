from jax.numpy import matmul
from pyneurons.functions import activation
from jax import jit


@jit
def positive(matrix, x):
    return activation(matmul(x, matrix))
