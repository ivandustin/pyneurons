from pyneurons.functions.core.activations import relu as function
from pyneurons.functions.vjps import identity

relu = identity(function)
