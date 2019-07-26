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
        '-' : 'line',    'o': 'circle', '△': 'triangle',
        '◇' : 'diamond', '□': 'square', '┸': 'quad',
        '+' : 'cross'
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
        for i in ('color', 'selection_color', 'nonselection_color'):
            color = self.__dict__.get(i, None)
            if isinstance(color, str) and len(color) and color[0] == '~':
                self.__dict__[i] = {'dark': f'light{color[1:]}', 'basic': f'dark{color[1:]}'}

def defaultfigsize(*args) -> Tuple[int, int, str]:
    "return the default fig size"
    return args+(700, 600, 'stretch_both')[len(args):] # type: ignore

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
    def __init__(self):
        self.theme   = deepcopy(self.theme)
        self.display = deepcopy(self.display)
        self.config  = deepcopy(self.config)
        assert self.theme.name
        assert self.display.name
        if self.config is not None:
            assert self.config.name, self
            assert self.config.name != self.theme.name, self

    def addto(self, ctrl, noerase = True):
        "sets-up model observers"
        self.theme   = ctrl.theme  .add(self.theme, noerase)
        self.display = ctrl.display.add(self.display, noerase)
        if self.config:
            self.config = ctrl.theme  .add(self.config, noerase)

    def observe(self, ctrl, noerase = False):
        "sets-up model observers"
        self.addto(ctrl, noerase)

    @classmethod
    def create(cls, ctrl, noerase = True):
        "creates the model and registers it"
        self = cls()
        self.addto(ctrl, noerase = noerase)
        return self
