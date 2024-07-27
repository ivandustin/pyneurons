from pytest import fixture
from jax.numpy import isclose
from pyneurons import PHI, bias


@fixture
def instance(key):
    return bias(key, shape=(1000,))


def test_mean(instance):
    assert isclose(instance.mean(), -PHI, atol=0.001)
