from jax.tree_util import register_pytree_node_class
from jax.numpy import negative
from pyneurons.functions import apply
from .neuron import Neuron


@register_pytree_node_class
class Inhibitory(Neuron):
    def __call__(self, x):
        return negative(apply(self.array, x))
