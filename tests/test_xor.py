from pytest import fixture
from multipledispatch import dispatch
from pyneurons.classes.tuples import Model
from pyneurons.classes.tuples.models import Neuron
from pyneurons.functions.pytree import concat
from pyneurons.functions.vjps import identity
from pyneurons.functions.subclassing.pytree import compose
from jax.numpy import array, mean, square, array_equal, heaviside, ndarray
from jax.tree_util import tree_map, register_pytree_node_class
from jax.lax import fori_loop
from jax.random import split
from jax import grad


@identity
def binary(x):
    return heaviside(x, 1)


Binary = compose("Binary", Neuron, binary)


@register_pytree_node_class
class XOR(Model):
    @dispatch(type, tuple)
    def __new__(cls, value):
        return super(Model, cls).__new__(cls, value)

    @dispatch(type, ndarray)
    def __new__(cls, key):
        key_a, key_b = split(key, 2)
        a = Binary(key_a, 2)
        b = Binary(key_b, 3)
        return cls.__new__(cls, (a, b))

    def __call__(self, x):
        a, b = self
        return b(concat([x, a(x)]))


@fixture
def x():
    return array([[0, 0], [0, 1], [1, 0], [1, 1]])


@fixture
def y():
    return array([[0], [1], [1], [0]])


@fixture
def model(key):
    return XOR(key)


def test(model, x, y):
    assert not array_equal(model(x), y)
    model = train(model, x, y)
    assert array_equal(model(x), y)


def loss(model, x, y):
    yhat = model(x)
    return mean(square(y - yhat))


def train(model, x, y):
    def body(_, model):
        gradient = grad(loss)(model, x, y)
        model = tree_map(lambda w, g: w - 0.1 * g, model, gradient)
        return model

    return fori_loop(0, 100, body, model)
