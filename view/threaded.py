#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Basics for threading displays"
import  asyncio
from    abc                 import abstractmethod
from    collections         import OrderedDict
from    contextlib          import contextmanager
from    enum                import Enum
from    importlib           import import_module
from    time                import time
from    typing              import TypeVar, Generic, ClassVar, Type, Optional, Callable

from    bokeh.document      import Document
from    bokeh.models        import Model

from    model.plots         import PlotAttrs
from    utils.logconfig     import getLogger
from    utils.inspection    import templateattribute as _tattr
from    .plots              import PlotAttrsView

LOGS     = getLogger(__name__)
MODEL    = TypeVar('MODEL', bound = 'DisplayModel')
DISPLAY  = TypeVar("DISPLAY")
THEME    = TypeVar("THEME")
ResetFcn = Optional[Callable[[dict], None]]

class DisplayState(Enum):
    "plot state"
    active       = 'active'
    abouttoreset = 'abouttoreset'
    resetting    = 'resetting'
    disabled     = 'disabled'
    outofdate    = 'outofdate'

class DisplayModel(Generic[DISPLAY, THEME]):
    "Basic model for time series"
    display: DISPLAY
    theme:   THEME

    def __init__(self, **kwa):
        super().__init__()

        def _create(ind):
            cls  = _tattr(self, ind)
            keys = getattr(cls, '__dataclass_fields__', kwa)
            return cls(**{i: j for i, j in kwa.items() if i in keys})

        self.display = _create(0)
        self.theme   = _create(1)

    def swapmodels(self, ctrl):
        "add the models to the controllers"
        if hasattr(self.theme, 'name'):
            ctrl.theme.swapmodels(self.theme)
        if hasattr(self.display, 'name'):
            ctrl.display.swapmodels(self.display)

        for i in self.__dict__.values():
            if callable(getattr(i, 'swapmodels', None)):
                i.swapmodels(ctrl)

    def observe(self, ctrl):
        "observes the model"
        for i in self.__dict__.values():
            if callable(getattr(i, 'observe', None)):
                i.observe(ctrl)

    def addto(self, ctrl):
        "swaps models & observes the controller"
        self.swapmodels(ctrl)
        self.observe(ctrl)

class _OrderedDict(OrderedDict):
    def __missing__(self, key):
        value: dict = OrderedDict()
        self[key]   = value
        return value

class _ResetterDescr:
    _name: str
    __slots__ = ('_name',)

    def __set_name__(self, _, name: str):
        self._name = name

    def __get__(self, inst, tpe):
        return self if inst is None else getattr(getattr(inst, '_view'), self._name)

    def __set__(self, inst, val):
        return setattr(getattr(inst, '_view'), self._name, val)

class _Resetter:
    _state = _ResetterDescr()
    _model = _ResetterDescr()
    _doc   = _ResetterDescr()
    _old: DisplayState

    def __init__(self, ctrl, view: 'ThreadedDisplay', fcn: ResetFcn):
        self._ctrl  = ctrl
        self._view  = view
        self._fcn   = fcn
        self._time  = [time()]*2
        self._cache = _OrderedDict()

    def __call__(self, nothreading: bool, ctx: bool = True):
        if getattr(BASE, 'SINGLE_THREAD') or nothreading:
            self.__reset(ctx)
            self._time[1] = time()
            self.__end()
            return

        self.__reset(False)
        self._old, self._state = self._state, DisplayState.abouttoreset
        asyncio.create_task(self._reset_and_render())

    def __end(self, other = None):
        name  = type(self._view).__qualname__
        delta = self._time[1] - self._time[0]
        self._ctrl.display.handle('rendered', args = {'element': self._view})
        if other:
            LOGS.debug("%s.reset done in %.3f+%.3f", name, delta, other)
        else:
            LOGS.debug("%s.reset done in %.3f", name, delta)

    def __reset(self, ctx: bool):
        mdl = getattr(self._model, 'reset', None)
        if not (mdl or ctx):
            return

        with self._view.resetting() as cache:
            if mdl:
                mdl(self._ctrl)
            if ctx:
                getattr(self._view, '_reset')(self._ctrl, cache)

    async def _reset_and_render(self):
        await asyncio.wrap_future(BASE.POOL.submit(self._reset_without_render))
        self._time[1] = time()
        if self._cache:
            self._doc.add_next_tick_callback(self._render)
        else:
            self.__end()

    def _reset_without_render(self):
        try:
            self._state = DisplayState.resetting
            with self._ctrl.computation.type(self._ctrl, calls = self.__call__):
                if self._fcn is None:
                    getattr(self._view, '_reset')(self._ctrl, self._cache)
                else:
                    self._fcn(self._cache)
        finally:
            self._state = self._old

    def _render(self):
        start = time()
        try:
            if self._cache:
                with self._ctrl.computation.type(self._ctrl, calls = self.__call__):
                    with self._view.resetting() as inp:
                        inp.update(self._cache)
        finally:
            self.__end(time() - start)

class ThreadedDisplay(Generic[MODEL]):  # pylint: disable=too-many-public-methods
    "Base plotter class"
    _doc:   Document
    _model: MODEL
    _state: DisplayState
    _RESET: ClassVar[Type[_Resetter]] = _Resetter

    def __init__(self, model: MODEL = None, **kwa) -> None:
        "sets up this plotter's info"
        super().__init__()
        self._model = _tattr(self, 0)(**kwa) if model is None else model
        self._state = DisplayState.active

    def swapmodels(self, ctrl):
        "swap models with those in the controller"
        for i in self.__dict__.values():
            if callable(getattr(i, 'swapmodels', None)):
                i.swapmodels(ctrl)

    def action(self, ctrl, fcn = None):
        "decorator which starts a user action but only if state is set to active"
        action = ctrl.action.type(
            ctrl,
            test = lambda *_1, **_2: self._state is DisplayState.active
        )
        return action if fcn is None else action(fcn)

    def delegatereset(self, ctrl, cache):
        "Stops on_change events for a time"
        old, self._state = self._state, DisplayState.resetting
        try:
            self._reset(ctrl, cache)
        finally:
            self._state     = old

    @contextmanager
    def resetting(self):
        "Stops on_change events for a time"
        mdls            = _OrderedDict()
        old, self._state = self._state, DisplayState.resetting
        i = j = None
        try:
            yield mdls
            for i, j in mdls.items():
                if isinstance(i, Model):
                    i.update(**j)
                elif callable(j):
                    j()
                else:
                    raise TypeError(f"No know way to update {i} = {j}")

        except ValueError as exc:
            if i is not None:
                raise ValueError(f'Error updating {i} = {j}') from exc
            raise ValueError(f'Error updating') from exc
        finally:
            self._state = old

    def close(self):
        "Removes the controller"
        del self._model
        if hasattr(self, '_ctrl'):
            delattr(self, '_ctrl')
        del self._doc

    def ismain(self, _):
        "Set-up things if this view is the main one"

    @staticmethod
    def attrs(attrs:PlotAttrs) -> PlotAttrsView:
        "shortcuts for PlotAttrsView"
        return PlotAttrsView(attrs)

    def addtodoc(self, ctrl, doc):
        "returns the figure"
        self._doc = doc
        with self.resetting():
            return self._addtodoc(ctrl, doc)

    def activate(self, ctrl, val, now = False):
        "activates the component: resets can occur"
        old        = self._state
        self._state = DisplayState.active if val else DisplayState.disabled
        if val and (old is DisplayState.outofdate):
            _Resetter(ctrl, self, None)(now)

    def reset(self, ctrl, now = False, fcn: ResetFcn = None):
        "Updates the data"
        state = self._state
        if   state is DisplayState.disabled:
            self._state = DisplayState.outofdate

        elif state is DisplayState.active:
            _Resetter(ctrl, self, fcn)(now)

        elif state is DisplayState.abouttoreset:
            _Resetter(ctrl, self, fcn)(True, False)

    def _waitfornextreset(self) -> bool:
        """
        can be used in observed events to tell whether to update the view
        or wait for the next update
        """
        if self._state == DisplayState.disabled:
            self._state = DisplayState.outofdate
            return True
        return self._state != DisplayState.active

    @abstractmethod
    def _addtodoc(self, ctrl, doc):
        "creates the plot structure"

    @abstractmethod
    def _reset(self, ctrl, cache):
        "initializes the plot for a new file"


BASE     = import_module(".base", package = __name__[:__name__.rfind('.')])
