from pyneurons.functions import apply
from .neuron import Neuron


class Excitatory(Neuron):
    def __call__(self, x):
        return apply(self.array, x)
