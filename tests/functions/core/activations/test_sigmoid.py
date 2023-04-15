from jax.numpy import array, array_equal, zeros_like
from pyneurons.functions.core.activations import sigmoid as function


def test_positive(positive):
    expected = array([0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert array_equal(function(positive), expected)


def test_negative(negative):
    assert array_equal(function(negative), zeros_like(negative))
