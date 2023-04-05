from pytest import fixture
from jax.numpy import array, array_equal, negative
from pyneurons.functions import apply
from pyneurons.classes import Inhibitory


@fixture
def neuron(key):
    return Inhibitory(key, 3)


@fixture
def x():
    return array([1.0, 2.0, 3.0])


@fixture
def y(neuron, x):
    return negative(apply(neuron.array, x))


def test_call(neuron, x, y):
    assert array_equal(neuron(x), y)
