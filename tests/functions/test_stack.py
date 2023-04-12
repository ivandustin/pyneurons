from pyneurons.classes.models import Neuron
from pyneurons.functions import stack


def test(key):
    assert stack([Neuron(key, 2)] * 2)[0].shape == (2, 2, 1)
