from pyneurons.functions.factories.models import subclass
from pyneurons.functions.activations import relu
from .neuron import Neuron

ReLU = subclass("ReLU", Neuron, relu)
