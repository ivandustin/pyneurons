from pyneurons.functions.factories.classes import compose
from pyneurons.functions.activations import binary
from pyneurons.classes import Neuron

Binary = compose(Neuron, binary, "Binary")
