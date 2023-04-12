from contextlib import suppress
from jax.tree_util import tree_flatten, tree_unflatten

map_function = map


def map(function, pytree):
    leaves, treedef = tree_flatten(pytree)
    entries = list(map_function(function, leaves))
    with suppress(IndexError):
        while True:
            leaves = list(map_function(lambda entry: entry.pop(0), entries))
            yield tree_unflatten(treedef, leaves)
