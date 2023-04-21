from pyneurons.concat import concat


def test(box):
    (array,) = concat([box] * 2)
    assert array.shape == (2, 6)
