from jax.numpy import ndarray
from jax.tree_util import register_pytree_node_class
from multipledispatch import dispatch


@register_pytree_node_class
class Array(tuple):
    @dispatch(type, ndarray)
    def __new__(cls, array):
        return super().__new__(cls, (array,))

    @property
    def array(self):
        return self[0]

    def tree_flatten(self):
        return self, None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)
