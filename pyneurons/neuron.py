from jax.numpy import ndarray
from multipledispatch import dispatch
from .random.weight import weight
from .random.bias import bias
from .random.key import key


@dispatch(ndarray, int)
def neuron(key, n):
    return (weight(key, (n, 1)), bias(key, (1, 1)))


@dispatch(int)
def neuron(n):
    return neuron(key(), n)
