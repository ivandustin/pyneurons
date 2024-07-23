from pytest import fixture
from jax.numpy import array, isclose
from pyneurons.fit import fit
from pyneurons import Neuron


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
    for learning_rate in [0.1, 0.01, 0.001]:
        for _ in range(10):
            model = fit(learning_rate, model, x, y)
    assert isclose(model(x), y)
