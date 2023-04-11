def compose(cls, function, name):
    def __call__(self, x):
        return function(cls.__call__(self, x))

    return type(name, (cls,), {"__call__": __call__})
