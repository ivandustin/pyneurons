from contextlib import suppress
from jax.tree_util import tree_flatten, tree_unflatten


def explode(function, pytree):
    leaves, treedef = tree_flatten(pytree)
    entries = list(map(function, leaves))
    with suppress(IndexError):
        while True:
            leaves = list(map(lambda entry: entry.pop(0), entries))
            yield tree_unflatten(treedef, leaves)
