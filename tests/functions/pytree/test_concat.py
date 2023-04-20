from pyneurons.classes.tuples.models import Neuron
from pyneurons.functions.pytree import concat


def test(key):
    assert concat([Neuron(key, 2)] * 2)[0].shape == (2, 2)
