from pytest import fixture
from jax.numpy import array as jax_array
from jax.random import PRNGKey
from pyneurons.box import Box


@fixture
def key():
    return PRNGKey(0)


@fixture
def array():
    return jax_array([[1, 2, 3], [4, 5, 6]])


@fixture
def box(array):
    return Box(array)
