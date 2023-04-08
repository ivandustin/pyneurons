from jax.tree_util import register_pytree_node_class
from jax.numpy import ndarray
from multipledispatch import dispatch
from pyneurons.functions import synapses
from .tuple import Tuple


@register_pytree_node_class
class Neuron(Tuple):
    @dispatch(type, ndarray)
    def __new__(cls, array):
        return super().__new__(cls, (array,))

    @dispatch(type, ndarray, int)
    def __new__(cls, key, x):
        return cls.__new__(cls, synapses(key, shape=(x, 1)))
