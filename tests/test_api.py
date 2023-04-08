import pyneurons as pn


def test_ex():
    assert pn.ex == pn.classes.Excitatory


def test_inh():
    assert pn.inh == pn.classes.Inhibitory


def test_concat():
    assert pn.concat == pn.functions.concat


def test_stack():
    assert pn.stack == pn.functions.stack
