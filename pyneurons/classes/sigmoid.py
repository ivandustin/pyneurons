from pyneurons.functions.factories.models import subclass
from pyneurons.functions.activations import sigmoid
from .neuron import Neuron

Sigmoid = subclass("Sigmoid", Neuron, sigmoid)
