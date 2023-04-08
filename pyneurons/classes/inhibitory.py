from jax.tree_util import register_pytree_node_class
from pyneurons.functions.apply import negative
from .neuron import Neuron


@register_pytree_node_class
class Inhibitory(Neuron):
    def __call__(self, x):
        return negative(self[0], x)
