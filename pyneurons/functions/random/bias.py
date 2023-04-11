from .param import param


def bias(key, shape):
    return param(key, shape) + 1.38196601125
