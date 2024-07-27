from jax.numpy import isclose
from pyneurons import PHI


def test():
    assert isclose(PHI, 1.618, atol=0.001)
