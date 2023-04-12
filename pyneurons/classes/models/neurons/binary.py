from pyneurons.functions.factories.classes.pytrees import compose
from pyneurons.functions.activations import binary
from pyneurons.classes.models import Neuron

Binary = compose(Neuron, binary, "Binary")
