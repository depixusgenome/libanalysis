#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"The basic architecture"
from copy   import deepcopy
from enum   import Enum
from typing import Tuple, Optional, Any, Dict

from utils  import initdefaults

RangeType  = Tuple[Optional[float], Optional[float]]

class PlotState(Enum):
    "plot state"
    active       = 'active'
    abouttoreset = 'abouttoreset'
    resetting    = 'resetting'
    disabled     = 'disabled'
    outofdate    = 'outofdate'

class PlotAttrs:
    "Plot Attributes for one variable"
    _GLYPHS = {
        '-': 'line',    'o': 'circle', '△': 'triangle',
        '◇': 'diamond', '□': 'square', '┸': 'quad',
        '+': 'cross'
    }

    def __init__(self,
                 color   = '~blue',
                 glyph   = 'line',
                 size    = 1,
                 palette = None,
                 **kwa) -> None:
        self.color   = color
        self.glyph   = self._GLYPHS.get(glyph, glyph)
        self.size    = size
        self.palette = palette
        self.__dict__.update(kwa)
        for i, j in list(self.__dict__.items()):
            if i.endswith('color') and isinstance(j, str) and j[:1] == '~':
                self.__dict__[i] = {'dark': f'light{j[1:]}', 'basic': f'dark{j[1:]}'}

def defaultfigsize(*args) -> Tuple[int, int, str]:
    "return the default fig size"
    return args+(700, 600, 'stretch_both')[len(args):]  # type: ignore

class PlotTheme:
    """
    Default plot theme
    """
    name:            str                  = ''
    ylabel:          str                  = 'Z (μm)'
    yrightlabel:     str                  = 'Bases'
    xtoplabel:       str                  = 'Time (s)'
    xlabel:          str                  = 'Frames'
    figsize:         Tuple[int, int, str] = defaultfigsize()
    overshoot:       float                = .001
    boundsovershoot: float                = 1.
    output_backend:  str                  = 'canvas'
    toolbar:         Dict[str, Any]       = dict(
        sticky   = False,
        location = 'above',
        items    = 'xpan,box_zoom,wheel_zoom,save',
        hide     = True
    )
    tooltips: Any                         = None
    @initdefaults(frozenset(locals()))
    def __init__(self, **kwa):
        pass

    defaultfigsize = staticmethod(defaultfigsize)

class PlotDisplay:
    """
    Default plot display
    """
    __NONE             = (None, None)
    name:    str       = ""
    state:   PlotState = PlotState.active
    xinit:   RangeType = __NONE
    yinit:   RangeType = __NONE
    xbounds: RangeType = __NONE
    ybounds: RangeType = __NONE

    @initdefaults(frozenset(locals()))
    def __init__(self, **kwa):
        pass

    def isactive(self) -> bool:
        "whether the plot is active"
        return self.state == PlotState.active

class PlotModel:
    """
    base plot model
    """
    theme:   PlotTheme   = PlotTheme()
    display: PlotDisplay = PlotDisplay()
    config:  Any         = None

    def __init__(self, **_):
        self.theme   = deepcopy(self.theme)
        self.display = deepcopy(self.display)
        self.config  = deepcopy(self.config)
        assert self.theme.name
        assert self.display.name
        if self.config is not None:
            assert self.config.name, self
            assert self.config.name != self.theme.name, self

    def swapmodels(self, ctrl):
        "sets-up model observers"
        self.theme   = ctrl.theme.swapmodels(self.theme)
        self.display = ctrl.display.swapmodels(self.display)
        if self.config:
            self.config = ctrl.theme.swapmodels(self.config)

    def observe(self, ctrl):
        "sets-up model observers"

    def addto(self, ctrl):
        "sets-up model observers"
        self.swapmodels(ctrl)
        self.observe(ctrl)
