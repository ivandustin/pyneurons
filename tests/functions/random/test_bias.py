from pytest import fixture
from jax.numpy import isclose
from pyneurons.functions.random import bias


@fixture
def instance(key):
    return bias(key, shape=(1000,))


def test_mean(instance):
    assert isclose(instance.mean(), 1.381, atol=0.01)
