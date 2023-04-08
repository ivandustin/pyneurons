from jax import jit
from jax.numpy import matmul
from jax.numpy import negative as negative_function
from pyneurons.functions import activation


@jit
def negative(matrix, x):
    return negative_function(activation(matmul(x, matrix)))
