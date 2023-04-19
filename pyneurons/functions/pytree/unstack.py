from pyneurons.functions import identity
from .explode import explode


def unstack(pytree):
    return list(explode(lambda array: list(map(identity, array)), pytree))
