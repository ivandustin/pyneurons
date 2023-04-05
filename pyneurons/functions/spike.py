from jax import custom_vjp
from .core.spike import spike as spike_function


spike = custom_vjp(spike_function)


def forward(x):
    return spike_function(x), None


def backward(_, gradient):
    return (gradient,)


spike.defvjp(forward, backward)
