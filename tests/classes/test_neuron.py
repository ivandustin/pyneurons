from pyneurons.classes import Neuron


def test_init(key):
    neuron = Neuron(key, 3)
    assert neuron.array.shape == (3, 1)


def test_init_y(key):
    neuron = Neuron(key, 3, 2)
    assert neuron.array.shape == (3, 2)
