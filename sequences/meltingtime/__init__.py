#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Means for computing melting times:
"""
from ._computations import TransitionStats
from ._old          import OldStatesTransitions
from ._data         import nndata
if __name__ == '__main__':
    from .__main__ import main
    main() # pylint: disable=no-value-for-parameter
