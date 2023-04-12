from pyneurons.functions.factories.classes.pytrees import compose
from pyneurons.functions.activations import binary
from .neuron import Neuron

Binary = compose("Binary", binary, Neuron)
