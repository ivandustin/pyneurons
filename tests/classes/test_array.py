from pytest import fixture
from jax.numpy import array
from pyneurons.classes import Array


@fixture
def ndarray():
    return array(1.0)


@fixture
def instance(ndarray):
    return Array(ndarray)


def test(instance, ndarray):
    assert instance.array == ndarray
