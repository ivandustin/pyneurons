from jax.numpy import array_equal
from pyneurons.split import split


def test(array):
    arrays = split(array)
    assert len(arrays) == 3
    assert arrays[0].shape == (2, 1)
    assert arrays[1].shape == (2, 1)
    assert arrays[2].shape == (2, 1)
    assert array_equal(arrays[0], array[:, 0:1])
    assert array_equal(arrays[1], array[:, 1:2])
    assert array_equal(arrays[2], array[:, 2:3])
