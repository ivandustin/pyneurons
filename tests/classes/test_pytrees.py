from pyneurons.classes.models import Neuron
from jax.tree_util import tree_map
from jax.numpy import array
from pytest import fixture


@fixture
def value():
    return tuple([array(1.0)] * 3)


@fixture(params=[Neuron])
def instance(request, value):
    return request.param(value)


def test_tree_map(instance):
    tree_map(lambda x: x + 1, instance)
