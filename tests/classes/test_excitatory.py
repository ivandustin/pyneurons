from pytest import fixture
from jax.numpy import array, array_equal
from pyneurons.functions.apply import positive
from pyneurons.classes import Excitatory


@fixture
def neuron(key):
    return Excitatory(key, 3)


@fixture
def x():
    return array([1.0, 2.0, 3.0])


@fixture
def expected(neuron, x):
    return positive(neuron[0], x)


def test_call(neuron, x, expected):
    assert array_equal(neuron(x), expected)
