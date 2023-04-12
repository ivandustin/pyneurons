from multipledispatch import dispatch
from jax.tree_util import register_pytree_node_class
from jax.numpy import ndarray
from jax.random import split
from pyneurons.functions.random import key as random_key
from pyneurons.functions.random import weight, bias
from ..model import Model


@register_pytree_node_class
class Neuron(Model):
    @dispatch(type, tuple)
    def __new__(cls, value):
        return super().__new__(cls, value)

    @dispatch(type, ndarray, int)
    def __new__(cls, key, x):
        kw, kb = split(key, 2)
        w = weight(kw, (x, 1))
        b = bias(kb, (1,))
        return cls.__new__(cls, (w, b))

    @dispatch(type, int)
    def __new__(cls, x):
        return cls.__new__(cls, random_key(), x)

    def __call__(self, x):
        w, b = self
        return (x @ w) + b
