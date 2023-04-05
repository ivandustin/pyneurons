from jax import custom_vjp
from .core.activation import activation as activation_function


activation = custom_vjp(activation_function)


def forward(x):
    return activation_function(x), None


def backward(_, gradient):
    return (gradient,)


activation.defvjp(forward, backward)
