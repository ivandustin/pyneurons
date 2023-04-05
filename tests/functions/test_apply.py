from pytest import fixture
from jax.numpy import array, matmul, array_equal
from pyneurons.functions import activation
from pyneurons.functions import apply


@fixture
def matrix():
    return array([[1.0, 2.0], [3.0, 4.0]])


@fixture
def x():
    return array([[1.0, 2.0], [3.0, 4.0]])


@fixture
def y(x, matrix):
    return activation(matmul(x, matrix))


def test(matrix, x, y):
    assert array_equal(apply(matrix, x), y)
