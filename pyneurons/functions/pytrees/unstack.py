from pyneurons.functions import identity
from .map import map as map_function


def unstack(pytree):
    return list(map_function(lambda array: list(map(identity, array)), pytree))
