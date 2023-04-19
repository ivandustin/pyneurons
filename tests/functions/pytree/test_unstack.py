from pytest import fixture
from jax.numpy import array, array_equal
from pyneurons.functions.pytree import unstack


@fixture
def input():
    return array([[1, 2, 3], [4, 5, 6]])


def test(input):
    arrays = unstack(input)
    assert len(arrays) == input.shape[0]
    for i in range(len(arrays)):
        assert array_equal(arrays[i], input[i])
