from jax.numpy import matmul
from .activation import activation


def apply(matrix, x):
    return activation(matmul(x, matrix))
