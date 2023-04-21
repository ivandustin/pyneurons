from . import random
from . import vjp
from .apply import apply
from .binary import binary
from .bind import bind
from .box import Box
from .brelu1 import brelu1
from .compose import compose
from .concat import concat
from .explode import explode
from .identity import identity
from .implode import implode
from .model import Model
from .neuron import neuron
from .relu import relu
from .relu1 import relu1
from .spark import spark
from .spike import spike
from .split import split
from .stack import stack
from .unstack import unstack

Neuron = bind("Neuron", neuron, apply)
Binary = compose("Binary", Neuron, binary)
BReLU1 = compose("BReLU1", Neuron, brelu1)
ReLU = compose("ReLU", Neuron, relu)
ReLU1 = compose("ReLU1", Neuron, relu1)
Spark = BReLU1
Spike = Binary

__all__ = [
    "random",
    "vjp",
    "apply",
    "binary",
    "bind",
    "Box",
    "brelu1",
    "compose",
    "concat",
    "explode",
    "identity",
    "implode",
    "Model",
    "neuron",
    "relu",
    "relu1",
    "spark",
    "spike",
    "split",
    "stack",
    "unstack",
    "Neuron",
    "Binary",
    "BReLU1",
    "ReLU",
    "ReLU1",
    "Spark",
    "Spike",
]
