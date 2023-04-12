from pyneurons.classes.models import Neuron
from pyneurons.functions import concat


def test(key):
    assert concat([Neuron(key, 2)] * 2)[0].shape == (2, 2)
