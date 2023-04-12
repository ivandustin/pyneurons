from functools import partial
from jax.numpy import concatenate
from .pytrees import reduce


def concat(pytrees):
    return reduce(partial(concatenate, axis=-1), pytrees)
