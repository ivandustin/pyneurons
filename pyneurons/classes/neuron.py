from multipledispatch import dispatch
from jax.tree_util import register_pytree_node_class
from jax.numpy import ndarray
from jax.random import split
from pyneurons.functions.random import key as random_key
from pyneurons.functions.random import params
from pyneurons.functions import apply
from .tuple import Tuple


@register_pytree_node_class
class Neuron(Tuple):
    @dispatch(type, tuple)
    def __new__(cls, value):
        return super().__new__(cls, value)

    @dispatch(type, ndarray, int)
    def __new__(cls, key, x):
        key_m, key_b, key_a = split(key, 3)
        m = params(key_m, (x, 1))
        b = params(key_b, (1,))
        a = params(key_a, (1,))
        return cls.__new__(cls, (m, b, a))

    @dispatch(type, int)
    def __new__(cls, x):
        return cls.__new__(cls, random_key(), x)

    def __call__(self, x):
        m, b, a = self
        return apply(m, b, a, x)
