from jax import grad
from jax.numpy import clip
from pyneurons.vjp.identity import identity


def test_function():
    assert grad(function)(0.0) != 1.0


def test_identity():
    assert grad(identity(function))(0.0) == 1.0


def function(x):
    return clip(x, 0, 1)
