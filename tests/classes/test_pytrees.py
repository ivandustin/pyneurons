from pytest import fixture
from jax.numpy import array
from jax.tree_util import tree_map
from pyneurons.classes import Neuron, Excitatory, Inhibitory


@fixture
def ndarray():
    return array(1.0)


@fixture(params=[Neuron, Excitatory, Inhibitory])
def instance(request, ndarray):
    return request.param(ndarray)


def test_tree_map(instance):
    tree_map(lambda x: x + 1, instance)
