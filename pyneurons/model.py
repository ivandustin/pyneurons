from .box import Box


class Model(Box):
    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and not kwargs and isinstance(args[0], Box):
            (box,) = args
            (pytree,) = box
        else:
            pytree = cls.constructor(*args, **kwargs)
        return super().__new__(cls, pytree)

    def __call__(self, *args, **kwargs):
        (pytree,) = self
        return self.__class__.apply(pytree, *args, **kwargs)

    def constructor(*args, **kwargs):
        pass

    def apply(*args, **kwargs):
        pass
