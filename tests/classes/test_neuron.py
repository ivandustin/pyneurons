from jax.numpy import array, array_equal
from pyneurons.functions import apply
from pyneurons.classes import Neuron


def test_init(key):
    neuron = Neuron(key, 3)
    assert neuron.array.shape == (3, 1)


def test_init_y(key):
    neuron = Neuron(key, 3, 2)
    assert neuron.array.shape == (3, 2)


def test_call(key):
    neuron = Neuron(key, 3)
    x = array([1.0, 2.0, 3.0])
    assert array_equal(neuron(x), apply(neuron.array, x))
