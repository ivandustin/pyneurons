from pyneurons.classes.models.neurons import Binary
from jax.numpy import array


def test(key):
    x = array([2.0])
    neuron = Binary(key, 1)
    assert neuron(x) == 1.0
