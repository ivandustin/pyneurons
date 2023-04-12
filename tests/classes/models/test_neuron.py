from pyneurons.classes.models import Neuron


def test_with_key(key):
    neuron = Neuron(key, 3)
    assert neuron[0].shape == (3, 1)
    assert neuron[1].shape == (1,)


def test_without_key():
    neuron = Neuron(3)
    assert neuron[0].shape == (3, 1)
    assert neuron[1].shape == (1,)
