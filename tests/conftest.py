from jax.random import PRNGKey
from pytest import fixture


@fixture
def key():
    return PRNGKey(0)
