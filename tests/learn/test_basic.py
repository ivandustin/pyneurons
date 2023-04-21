from pytest import fixture
from pyneurons import Neuron
from pyneurons.fit import fit
from jax.numpy import array, isclose


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
    for i in range(23):
        model = fit(model, x, y)
    assert isclose(model(x), y)
