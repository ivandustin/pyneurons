from .weight import weight
from .bias import bias


def create(key, n):
    w = weight(key, shape=(n, n))
    b = bias(key, shape=(n, 1))
    return (w, b)
