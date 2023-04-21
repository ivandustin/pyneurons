from jax import jit


def compose(name, cls, function):
    def __call__(self, x):
        return jit(function)(cls.__call__(self, x))

    return type(name, (cls,), {"__call__": __call__})
