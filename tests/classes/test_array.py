from jax.tree_util import tree_map
from jax.numpy import array
from pyneurons.classes.array import Array


def test():
    a = Array(array(1.0))
    b = tree_map(lambda x: x + 1, a)
    assert b.array == 2.0


def test_str():
    a = Array(0)
    assert str(a) == "Array(0)"


def test_repr():
    a = Array(0)
    assert repr(a) == str(a)
