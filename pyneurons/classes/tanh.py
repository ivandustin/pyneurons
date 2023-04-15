from pyneurons.functions.factories.models import subclass
from pyneurons.functions.activations import tanh
from .neuron import Neuron

Tanh = subclass("Tanh", Neuron, tanh)
