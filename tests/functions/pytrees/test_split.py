from pytest import fixture
from pyneurons.functions.pytrees import split
from jax.numpy import array, array_equal


@fixture
def input():
    return array([[1, 2], [3, 4], [5, 6]])


def test(input):
    arrays = split(input)
    assert len(arrays) == 2
    assert arrays[0].shape == (3, 1)
    assert arrays[1].shape == (3, 1)
    assert array_equal(arrays[0], input[:, 0:1])
    assert array_equal(arrays[1], input[:, 1:2])
