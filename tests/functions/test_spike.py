from functools import partial
from jax.numpy import array, array_equal, inf, nan, ones_like
from jax import grad
from pytest import fixture
from pyneurons.functions.core import spike as spike_function
from pyneurons.functions import spike


@fixture
def input():
    return array([nan, -inf, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, inf])


def test(input):
    assert array_equal(spike(input), spike_function(input))


def test_grad_spike(input):
    assert array_equal(grad(partial(loss, spike))(input), ones_like(input))


def test_grad_spike_function(input):
    assert not array_equal(grad(partial(loss, spike_function))(input), ones_like(input))


def loss(function, input):
    return function(input).sum()
