#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Cycles plot view for cleaning data"
from typing         import (
    Dict, Any, Union, List, Tuple, Set, Iterator, overload, cast
)
from bokeh.colors   import named    as _bkclr
from bokeh          import palettes as _palette
import numpy as np


_bkclr.dpxblue = _bkclr.NamedColor("dpxblue", *(int('#6baed6'[i:i+2], 16) for i in (1, 3, 5)))


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

def tohex(clr:Union[Dict[Any,str], List[str], Tuple[str], Set[str], str, None]):
    "return the hex value"
    return (
        None                                if clr is None             else

        type(cast(dict, clr))((i, tohex(j)) for i, j in getattr(clr, 'items')())
        if callable(getattr(clr, 'items', None)) else

        (tohex(j) for j in clr)         if isinstance(clr, Iterator)   else

        type(cast(list, clr))(tohex(j) for j in clr)
        if not isinstance(clr, str)   else

        clr                             if len(clr) and clr[0] == '#' else
        getattr(_bkclr, clr).to_hex()   if hasattr(_bkclr, clr)       else
        getattr(_palette, clr)
    )

def palette(name: str, values) -> Dict[Any, str]:
    "return the best possible palette"
    if callable(getattr(_palette, name.lower(), None)):
        return dict(zip(values, getattr(_palette, name.lower())(len(values))))

    vals = getattr(_palette, name, 'Blues')
    if isinstance(vals, dict):
        vals = next((j for i, j in vals.items() if len(values) <= i), vals[max(vals)])

    if len(values) == len(vals):
        return dict(zip(values, vals))
    if len(values) < len(vals):
        # pylint: disable=no-member
        return dict(zip(values, _palette.linear_palette(vals, len(values))))
    return dict(zip(
        values,
        [vals[i] for i in np.linspace(0, len(vals)-1, len(values), dtype = 'i8')]
    ))
