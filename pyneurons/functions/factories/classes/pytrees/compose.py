from pyneurons.functions.factories.classes import compose as compose_function
from jax.tree_util import register_pytree_node_class


def compose(*args, **kwargs):
    return register_pytree_node_class(compose_function(*args, **kwargs))
