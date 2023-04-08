from pytest import fixture
from jax.numpy import matmul
from jax.numpy import array_equal, negative as negative_function
from pyneurons.functions.apply import negative
from pyneurons.functions import activation


@fixture
def expected(x, matrix):
    return negative_function(activation(matmul(x, matrix)))


def test(matrix, x, expected):
    assert array_equal(negative(matrix, x), expected)
