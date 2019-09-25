#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"utils for inspecting objects and frames"
from   typing      import Dict, Any
from   collections import ChainMap
from   .inspection import diffobj


class ConfigObject:
    """
    Object with a few helper function for comparison
    """
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __init_subclass__(cls, hashattributes = (), **_):
        if hashattributes:
            def __hash__(self):
                return hash((self.__class__, *(getattr(self, i, '?') for i in hashattributes)))
            cls.__hash__ = __hash__
        return super().__init_subclass__(**_)

    def diff(self, other) -> Dict[str, Any]:
        "return the diff with `other`"
        return diffobj(self, other)

    def config(self, tpe = dict):
        "return a chainmap with default and updated values"
        fcn   = getattr(self, '__getstate__', None)
        # pylint: disable=not-callable
        dself = fcn() if callable(fcn) else dict(self.__dict__)
        if tpe in (dict, 'dict'):
            return dself

        if all(hasattr(self.__class__, i) for i in dself):
            other = {i: getattr(self.__class__, i) for i in dself}
        else:
            # don't create a new instance unless necessary
            other = self.__class__().config(dict)
        return ChainMap(diffobj(dself, other), other)

def bind(ctrl, master, slave):
    """
    bind to main tasks model
    """
    if isinstance(master, str):
        master = ctrl.model(master)

    if isinstance(slave, str):
        slave = ctrl.model(slave)

    if master is not slave:
        ctrl.observe(master, lambda **_: ctrl.update(slave, **slave.diff(master)))
