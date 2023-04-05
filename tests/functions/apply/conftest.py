from jax.numpy import array
from pytest import fixture


@fixture
def matrix():
    return array([[1.0, 2.0], [3.0, 4.0]])


@fixture
def x():
    return array([[1.0, 2.0], [3.0, 4.0]])
