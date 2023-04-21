from pytest import fixture
from pyneurons import Neuron
from jax.numpy import array, isclose
from .functions import train


@fixture
def x():
    return array([1.0])


@fixture
def y():
    return array([1.7])


@fixture
def model(key):
    return Neuron(key, 1)


def test(model, x, y):
    assert not isclose(model(x), y)
    model = train(model, x, y, epochs=50)
    assert isclose(model(x), y)
