from pyneurons.functions.pytree import stack
from pyneurons.classes import Neuron


def test(key):
    assert stack([Neuron(key, 2)] * 2)[0].shape == (2, 2, 1)
