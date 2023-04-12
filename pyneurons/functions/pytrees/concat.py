from functools import partial
from jax.numpy import concatenate
from .reduce import reduce


def concat(pytrees):
    return reduce(partial(concatenate, axis=-1), pytrees)
