from jax.random import PRNGKey
from .randseed import randseed


def randkey():
    return PRNGKey(randseed())
