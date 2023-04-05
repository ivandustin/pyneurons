from jax.numpy import matmul
from pyneurons.functions import activation


def positive(matrix, x):
    return activation(matmul(x, matrix))
