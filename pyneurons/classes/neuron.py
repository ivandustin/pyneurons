from pyneurons.functions import synapse, apply
from .array import Array


class Neuron(Array):
    def __init__(self, key, x, y=1):
        super().__init__(synapse(key, shape=(x, y)))

    def __call__(self, x):
        return apply(self.array, x)
