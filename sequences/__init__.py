#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"All sequences-related stuff"
from    enum        import Enum
from   .io          import read
from   .translator  import (
    PEAKS_DTYPE, PEAKS_TYPE, Translator, peaks, splitoligos,
    complement, reversecomplement, gccontent,
    marksequence, markedoligos, overlap
)

class Strand(Enum):
    """
    Which strand we sit on
    """
    positive = True
    negative = False

    @classmethod
    def _missing_(cls, value):
        return getattr(cls, value)
