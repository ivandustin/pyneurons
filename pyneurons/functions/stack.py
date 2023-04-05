from jax.numpy import stack as jnp_stack
from jax.tree_util import tree_map


def stack(sequence):
    return tree_map(lambda *args: jnp_stack(args), *sequence)
