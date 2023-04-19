from pytest import fixture
from pyneurons.classes import Neuron
from jax.numpy import array, mean, square, isclose
from jax.tree_util import tree_map
from jax.lax import fori_loop
from jax import grad


@fixture
def x():
    return array([1.0])


@fixture
def y():
    return array([1.7])


@fixture
def neuron(key):
    return Neuron(key, 1)


def test(neuron, x, y):
    assert not isclose(neuron(x), y)
    neuron = train(neuron, x, y)
    assert isclose(neuron(x), y)


def loss(neuron, x, y):
    yhat = neuron(x)
    return mean(square(y - yhat))


def train(neuron, x, y):
    def body(_, neuron):
        gradient = grad(loss)(neuron, x, y)
        return tree_map(lambda w, g: w - 0.1 * g, neuron, gradient)

    return fori_loop(0, 50, body, neuron)
