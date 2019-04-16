#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class with means for rescaling some attributes
"""
from abc        import ABC
from typing     import Tuple, TypeVar, Iterator, Any
from copy       import deepcopy

def addzscaledattributes(cls, zattributes = ()):
    "add the zscaledattributes function"
    if zattributes:
        itms = tuple(
            i               if isinstance(i, tuple) else
            (i[1:], -1)     if i[0] == '~'          else
            (i, 1)
            for i in zattributes
        )

        def zscaledattributes() -> Tuple[Tuple[str, int],...]:
            "return the names of attributes scaled to Z"
            return itms
        cls.zscaledattributes = staticmethod(zscaledattributes)

SELF = TypeVar('SELF')
def rescale(self: SELF, value:float) -> SELF:
    "rescale factors (from µm to V for example) for a given bead"
    vals = getattr(self, 'zscaled')(value)
    if hasattr(self, '__getstate__'):
        state = getattr(self, '__getstate__')()
        state.update(vals)

        cpy   = type(self).__new__(type(self))
        getattr(cpy, '__setstate__')(deepcopy(state))
    else:
        cpy  = deepcopy(self)
        for i, j in vals:
            setattr(cpy, i, j)
    return cpy

def zscaled(self, value:float) -> Iterator[Tuple[str, Any]]:
    "return the rescaled attributes scaled"
    def _rescale(old, factor):
        if old is None or isinstance(old, str):
            return old

        if isinstance(old, (list, set, tuple)):
            return type(old)(_rescale(i, factor) for i in old)

        if isinstance(old, dict):
            return type(old)(
                (_rescale(i, factor), _rescale(j, factor))
                for i, j in old.items()
            )

        if hasattr(old, 'rescale'):
            assert factor == 1.
            return old.rescale(value)
        return old*pow(value, factor)

    if not hasattr(self, '__getstate__'):
        return ((i, _rescale(getattr(self, i), j)) for i, j in self.zscaledattributes())

    attrs = dict(self.zscaledattributes())
    return (
        (i, _rescale(j, attrs[i]))
        for i, j in getattr(self, '__getstate__')().items()
        if i in attrs
    )

class Rescaler:
    "Class for rescaling z-axis-dependant attributes"
    def __init_subclass__(cls, zattributes = (), **kwa):
        addzscaledattributes(cls, zattributes)
        super().__init_subclass__(**kwa)

    def rescale(self, value:float) -> 'Rescaler':
        "rescale factors (from µm to V for example) for a given bead"
        return rescale(self, value)

    def zscaled(self, value:float) -> Iterator[Tuple[str, Any]]:
        "return the rescaled attributes scaled"
        return zscaled(self, value)

    @staticmethod
    def zscaledattributes() -> Tuple[str,...]:
        "return the names of attributes scaled to Z"
        return ()

class ARescaler(ABC):
    "Class for rescaling z-axis-dependant attributes"
    def __init_subclass__(cls, zattributes = (), **kwa):
        addzscaledattributes(cls, zattributes)
        super().__init_subclass__(**kwa)

    def rescale(self, value:float) -> 'ARescaler':
        "rescale factors (from µm to V for example) for a given bead"
        return rescale(self, value)

    def zscaled(self, value:float) -> Iterator[Tuple[str, Any]]:
        "return the rescaled attributes scaled"
        return zscaled(self, value)

    @staticmethod
    def zscaledattributes() -> Tuple[str,...]:
        "return the names of attributes scaled to Z"
        return ()
