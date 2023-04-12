from pyneurons.functions.pytrees import concat
from pyneurons.classes import Neuron


def test(key):
    assert concat([Neuron(key, 2)] * 2)[0].shape == (2, 2)
