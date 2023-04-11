from .param import param


def weight(key, shape):
    return param(key, shape) + 1.5
