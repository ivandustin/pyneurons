from jax.tree_util import register_pytree_node_class
from pyneurons.functions import apply
from .neuron import Neuron


@register_pytree_node_class
class Excitatory(Neuron):
    def __call__(self, x):
        return apply(self.array, x)
