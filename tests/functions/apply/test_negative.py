from pytest import fixture
from jax.numpy import array_equal, negative as negative_function
from pyneurons.functions.apply import positive, negative


@fixture
def expected(matrix, x):
    return negative_function(positive(matrix, x))


def test(matrix, x, expected):
    assert array_equal(negative(matrix, x), expected)
