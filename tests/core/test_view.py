#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Testing views"
from typing       import Iterator
from bokeh.colors import named    as _bkclr
from bokeh        import palettes as _palette
from view.colors  import tohex

class _dico:
    def __init__(self, *vals, **kwa):
        self.__dict__.update(*vals, **kwa)

    def items(self):
        "get items"
        return self.__dict__.items()

def test_colors():
    "test color retrieval"
    assert tohex(None) is None
    assert tohex('gray') == _bkclr.gray.to_hex()
    assert tohex('grey') == _bkclr.gray.to_hex()
    assert tohex('red') == _bkclr.red.to_hex()
    assert tohex('RdYlGn9') == getattr(_palette, 'RdYlGn9')
    assert tohex(['brown']) == [_bkclr.brown.to_hex()]
    assert tohex({'brown'}) == {_bkclr.brown.to_hex()}
    assert tohex(('brown',)) == (_bkclr.brown.to_hex(),)
    assert isinstance(tohex(iter(('brown',))), Iterator)
    assert isinstance(tohex(_dico()), _dico)
    assert dict(tohex(_dico(x = 'lightblue')).items()) == {'x': _bkclr.lightblue.to_hex()}

if __name__ == '__main__':
    test_colors()
