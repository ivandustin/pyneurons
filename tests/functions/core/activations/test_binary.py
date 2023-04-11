from pytest import fixture
from jax.numpy import array, array_equal, inf, nan
from pyneurons.functions.core.activations import binary


@fixture
def input():
    return array([nan, -inf, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, inf])


@fixture
def expected():
    return array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def test(input, expected):
    assert array_equal(binary(input), expected)
