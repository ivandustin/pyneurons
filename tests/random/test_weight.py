from pytest import fixture
from jax.numpy import isclose
from pyneurons import PHI, weight


@fixture
def instance(key):
    return weight(key, shape=(1000,))


def test_mean(instance):
    assert isclose(instance.mean(), PHI, atol=0.001)
