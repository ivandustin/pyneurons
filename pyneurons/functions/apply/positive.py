from jax import jit
from jax.numpy import matmul
from pyneurons.functions import activation


@jit
def positive(matrix, x):
    return activation(matmul(x, matrix))
