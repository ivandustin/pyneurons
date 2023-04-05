from pyneurons.functions import concat
from pyneurons.classes import Neuron


def test(key):
    assert concat([Neuron(key, 2)] * 2).array.shape == (2, 2)
