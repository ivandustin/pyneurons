import pyneurons as pn
from pytest import fixture
from jax.numpy import array, mean, square, array_equal
from jax.tree_util import tree_map
from jax.lax import fori_loop
from jax.random import split
from jax import grad


@fixture
def x():
    return array([[0, 0], [0, 1], [1, 0], [1, 1]])


@fixture
def y():
    return array([[0], [1], [1], [0]])


@fixture
def model(key):
    key_a, key_b = split(key, 2)
    a = pn.Binary(key_a, 2)
    b = pn.Binary(key_b, 3)
    return (a, b)


def test(model, x, y):
    assert not array_equal(apply(model, x), y)
    model = train(model, x, y)
    assert array_equal(apply(model, x), y)


def apply(model, x):
    a, b = model
    return b(pn.concat([x, a(x)]))


def loss(model, x, y):
    yhat = apply(model, x)
    return mean(square(y - yhat))


def train(model, x, y):
    def body(_, model):
        gradient = grad(loss)(model, x, y)
        model = tree_map(lambda w, g: w - 0.1 * g, model, gradient)
        return model

    return fori_loop(0, 100, body, model)
