from jax.numpy import negative
from pyneurons.functions import apply
from .neuron import Neuron


class Inhibitory(Neuron):
    def __call__(self, x):
        return negative(apply(self.array, x))
