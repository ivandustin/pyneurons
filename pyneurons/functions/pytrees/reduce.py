from jax.tree_util import tree_map


def reduce(function, pytrees):
    return tree_map(lambda *leaves: function(list(leaves)), *pytrees)
