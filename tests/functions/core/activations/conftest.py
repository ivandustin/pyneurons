from jax.numpy import array, inf
from pytest import fixture


@fixture
def positive():
    return array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, inf])


@fixture
def negative(positive):
    return -positive
