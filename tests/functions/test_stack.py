from pyneurons.functions import stack
from pyneurons.classes import Neuron


def test(key):
    assert stack([Neuron(key, 2)] * 2).array.shape == (2, 2, 1)
