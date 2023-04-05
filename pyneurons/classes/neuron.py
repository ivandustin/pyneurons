from pyneurons.functions import synapse
from .array import Array


class Neuron(Array):
    def __init__(self, key, x, y=1):
        super().__init__(synapse(key, shape=(x, y)))
