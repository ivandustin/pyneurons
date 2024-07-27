from pytest import fixture
from jax.numpy import float32, isclose
from pyneurons import param


@fixture
def instance(key):
    return param(key, shape=(1000,))


def test_dtype(instance):
    assert instance.dtype == float32


def test_type(instance):
    assert not instance.weak_type


def test_mean(instance):
    assert isclose(instance.mean(), 0.0, atol=0.01)


def test_std(instance):
    assert isclose(instance.std(), 0.1, atol=0.01)
