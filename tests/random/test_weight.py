from pytest import fixture
from jax.numpy import isclose
from pyneurons.random.weight import weight


@fixture
def instance(key):
    return weight(key, shape=(1000,))


def test_mean(instance):
    assert isclose(instance.mean(), 1.618, atol=0.01)
