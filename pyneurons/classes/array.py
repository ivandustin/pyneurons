from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Array:
    def __init__(self, array):
        self.array = array

    def __str__(self):
        return f"Array({self.array})"

    def __repr__(self):
        return str(self)

    def tree_flatten(self):
        return (self.array,), None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)
