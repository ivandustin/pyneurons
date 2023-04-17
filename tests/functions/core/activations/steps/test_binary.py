from pytest import fixture
from jax.numpy import array, array_equal
from pyneurons.functions.core.activations.steps import binary


@fixture
def input():
    return array([-0.5, 0, 0.5])


@fixture
def expected():
    return array([0, 1, 1])


def test(input, expected):
    assert array_equal(binary(input), expected)
