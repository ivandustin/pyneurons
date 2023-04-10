from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Tuple(tuple):
    def tree_flatten(self):
        return self, None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(children)
