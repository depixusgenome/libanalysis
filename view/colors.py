#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Cycles plot view for cleaning data"
from typing         import (
    Dict, Any, Union, List, Tuple, Set, Iterator, overload, cast
)
from bokeh.colors   import named    as _bkclr
from bokeh          import palettes as _palette

@overload
def tohex(clr:Dict[Any,str]) -> Dict[Any, str]:
    "return a dictionnary of colors in hex format"
@overload
def tohex(clr:Tuple[str]) -> Tuple[str]:
    "return an iteration of colors in hex format"
@overload
def tohex(clr:List[str]) -> List[str]:
    "return an iteration of colors in hex format"
@overload
def tohex(clr:Set[str]) -> Set[str]:
    "return an iteration of colors in hex format"
@overload
def tohex(clr:str) -> Union[str, List[str]]:
    "return either the palette or the hex color associated to a name"

def tohex(clr:Union[Dict[Any,str], List[str], Tuple[str], Set[str], str]):
    "return the hex value"
    return (
        type(cast(dict, clr))((i, tohex(j)) for i, j in getattr(clr, 'items')())
        if callable(getattr(clr, 'items', None)) else

        (tohex(j) for j in clr)         if isinstance(clr, Iterator)   else

        type(cast(list, clr))(tohex(j) for j in clr)
        if not isinstance(clr, str)   else

        clr                             if len(clr) and clr[0] == '#' else
        getattr(_palette, clr)          if hasattr(_palette, clr)     else
        getattr(_bkclr, clr).to_hex()
    )
