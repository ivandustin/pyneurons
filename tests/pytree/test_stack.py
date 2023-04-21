from pyneurons.stack import stack


def test(box):
    (array,) = stack([box] * 2)
    assert array.shape == (2, 2, 3)
