from jax.tree_util import register_pytree_node_class
from jax.numpy import ndarray
from multipledispatch import dispatch
from pyneurons.functions import synapse
from .array import Array


@register_pytree_node_class
class Neuron(Array):
    @dispatch(type, ndarray, int)
    def __new__(cls, key, x):
        return super().__new__(cls, synapse(key, shape=(x, 1)))
