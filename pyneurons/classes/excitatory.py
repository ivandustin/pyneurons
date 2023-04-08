from jax.tree_util import register_pytree_node_class
from pyneurons.functions.apply import positive
from .neuron import Neuron


@register_pytree_node_class
class Excitatory(Neuron):
    def __call__(self, x):
        return positive(self[0], x)
