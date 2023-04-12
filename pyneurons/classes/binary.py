from pyneurons.functions.factories.models import subclass
from pyneurons.functions.activations import binary
from .neuron import Neuron

Binary = subclass("Binary", Neuron, binary)
