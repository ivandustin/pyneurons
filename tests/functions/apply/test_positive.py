from pytest import fixture
from jax.numpy import matmul, array_equal
from pyneurons.functions.apply import positive
from pyneurons.functions import activation


@fixture
def expected(x, matrix):
    return activation(matmul(x, matrix))


def test(matrix, x, expected):
    assert array_equal(positive(matrix, x), expected)
