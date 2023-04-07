from pyneurons.functions import compose, identity


def test_order():
    assert compose(lambda x: x * 2, lambda x: x + 1)(1) == 4


def test_multiple():
    assert compose(identity, identity, identity)(1) == 1
