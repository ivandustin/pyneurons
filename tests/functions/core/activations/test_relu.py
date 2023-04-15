from jax.numpy import array, array_equal, zeros_like, inf
from pyneurons.functions.core.activations import relu as function


def test_positive(positive):
    expected = array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, inf])
    assert array_equal(function(positive), expected)


def test_negative(negative):
    assert array_equal(function(negative), zeros_like(negative))
