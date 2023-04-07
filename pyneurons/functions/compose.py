from functools import reduce
from .identity import identity


def compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, identity)
