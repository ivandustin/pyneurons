from pytest import fixture
from jax.numpy import array, array_equal
from pyneurons.functions.core.activations import tanh as function


@fixture
def expected():
    return array([0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def test_positive(positive, expected):
    assert array_equal(function(positive), expected)


def test_negative(negative, expected):
    assert array_equal(function(negative), -expected)
