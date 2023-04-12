from jax.numpy import stack as jnp_stack
from .pytrees import reduce


def stack(pytrees):
    return reduce(jnp_stack, pytrees)
