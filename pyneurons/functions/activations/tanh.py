from pyneurons.functions.core.activations import tanh as function
from pyneurons.functions.vjps import identity

tanh = identity(function)
