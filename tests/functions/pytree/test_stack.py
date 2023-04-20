from pyneurons.classes.tuples.models import Neuron
from pyneurons.functions.pytree import stack


def test(key):
    assert stack([Neuron(key, 2)] * 2)[0].shape == (2, 2, 1)
