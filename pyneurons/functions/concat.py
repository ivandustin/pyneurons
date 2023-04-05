from jax.tree_util import tree_map
from jax.numpy import concatenate


def concat(sequence):
    return tree_map(lambda *args: concatenate(args, axis=-1), *sequence)
