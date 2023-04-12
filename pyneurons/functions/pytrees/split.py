from jax.numpy import split as split_function
from .map import map as map_function


def split(pytree):
    return list(
        map_function(
            lambda array: split_function(array, array.shape[-1], axis=-1), pytree
        )
    )
