#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Utils for removing our modules views"
from   itertools import product
from   typing    import Sequence, Any, Tuple
import sys
import os

def modulecleanup(
        *names: str,
        pairs: Sequence[Tuple[str, Any]] = ()):
    """
    removes all modules with names in their path
    """
    def _cleanup():
        conda = 'miniconda3'
        root  = str(os.__file__)
        if conda in root:
            root = root[:root.find(conda)+len(conda)+1]
        else:
            root = root[:root.rfind('/')]
            root = root[:root.rfind('/')]

        for key, _   in pairs:
            sys.modules.pop(key, None)
        for key, mod in list(sys.modules.items()):
            path = getattr(mod, "__path__", [])
            if not path:
                path = [getattr(mod, "__file__", "miniconda3")]
            if any(i.startswith(root) for i in path):
                continue
            if any(i in j for i, j in product(names, path)):
                sys.modules.pop(key, None)
    try:
        _cleanup()
        for i, j in pairs:
            sys.modules[i] = j
        yield
    finally:
        _cleanup()
