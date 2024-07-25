from pytest import fixture
from jax.numpy import array, array_equal
from jax.random import split
from pyneurons.concat import concat
from pyneurons.bind import bind
from pyneurons.fit import fit
from pyneurons import Binary


@fixture
def x():
    return array([[0, 0], [0, 1], [1, 0], [1, 1]])


@fixture
def y():
    return array([[0], [1], [1], [0]])


@fixture
def XOR():
    return bind("XOR", create, apply)


@fixture
def model(XOR, key):
    return XOR(key)


def test(model, x, y):
    assert not array_equal(model(x), y)
    for _ in range(100):
        model = fit(0.1, model, x, y)
    assert array_equal(model(x), y)


def create(key):
    key_a, key_b = split(key, 2)
    a = Binary(key_a, 2)
    b = Binary(key_b, 3)
    return (a, b)


def apply(model, x):
    a, b = model
    return b(concat([x, a(x)]))
