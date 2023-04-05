from jax.tree_util import register_pytree_node_class
from jax.numpy import ndarray
from multipledispatch import dispatch
from pyneurons.functions import synapse
from .array import Array


@register_pytree_node_class
class Neuron(Array):
    @dispatch(ndarray)
    def __init__(self, array):
        super().__init__(array)

    @dispatch(ndarray, int, int)
    def __init__(self, key, x, y):
        super().__init__(synapse(key, shape=(x, y)))

    @dispatch(ndarray, int)
    def __init__(self, key, x):
        self.__init__(key, x, 1)
