from jax.numpy import ones, array_equal
from pyneurons.classes import Neuron


def test_init(key):
    neuron = Neuron(key, 3)
    assert neuron[0].shape == (3, 1)


def test_init_array():
    array = ones(shape=(3, 2))
    neuron = Neuron(array)
    assert array_equal(neuron[0], array)


def test_random_key():
    neuron = Neuron(3)
    assert neuron[0].shape == (3, 1)
