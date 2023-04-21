from jax.numpy import array_equal
from pyneurons.unstack import unstack


def test(array):
    arrays = unstack(array)
    assert len(arrays) == array.shape[0]
    for i in range(len(arrays)):
        assert array_equal(arrays[i], array[i])
