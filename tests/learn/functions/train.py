from jax import grad
from jax.lax import fori_loop
from jax.tree_util import tree_map
from .loss import loss


def train(model, x, y, epochs):
    def body(_, model):
        gradient = grad(loss)(model, x, y)
        return tree_map(lambda w, g: w - 0.1 * g, model, gradient)

    return fori_loop(0, epochs, body, model)
