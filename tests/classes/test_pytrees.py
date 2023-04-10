from pytest import fixture
from jax.numpy import array
from jax.tree_util import tree_map
from pyneurons.classes import Tuple, Neuron


@fixture
def value():
    return tuple([array(1.0)] * 3)


@fixture(params=[Tuple, Neuron])
def instance(request, value):
    return request.param(value)


def test_tree_map(instance):
    tree_map(lambda x: x + 1, instance)
