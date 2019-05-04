#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"all view aspects here"
from collections         import OrderedDict
from pathlib             import Path
from typing              import Dict, ClassVar, TypeVar, Tuple, Generic, Type

from bokeh               import layouts
from bokeh.models        import Panel, Spacer, Tabs

from utils.inspection    import templateattribute
from model.plots         import PlotState
from modaldialog         import dialog
from modaldialog.builder import changelog
from view.base           import BokehView
from version             import timestamp as _timestamp

class TabsTheme:
    "Tabs Theme"
    CHANGELOG = "CHANGELOG.html"
    def __init__(self, initial: str, panels: Dict[Type, str], version = 0, startup = "") -> None:
        self.name:    str            = "app.tabs"
        self.startup: str            = startup
        self.initial: str            = initial
        self.width:   int            = 1000
        self.height:  int            = 30
        self.titles:  Dict[str, str] = OrderedDict()
        for i, j in panels.items():
            self.titles[j] = getattr(i, 'PANEL_NAME', j.capitalize())
        assert self.initial in self.titles
        self.version: int = _timestamp() if version == 0 else version

    @classmethod
    def defaultstartup(cls, name):
        "extracts default startup message from a changelog"
        path = Path(".").absolute()
        for _ in range(4):
            if (path/cls.CHANGELOG).exists():
                break
            path = path.parent
        else:
            return None

        path /= cls.CHANGELOG
        with open(path, "r", encoding="utf-8") as stream:
            return changelog(stream, name)
        return None

TThemeType = TypeVar("TThemeType", bound = TabsTheme)
class TabsView(Generic[TThemeType], BokehView):
    "A view with all plots"
    KEYS    : ClassVar[Dict[type, str]]
    NAME    : ClassVar[str]
    _tabs   : Tabs
    __theme : TabsTheme
    def __init__(self, ctrl = None, **kwa):
        "Sets up the controller"
        super().__init__(ctrl = ctrl, **kwa)
        mdl           = templateattribute(self, 0)() # type: ignore
        self.__theme  = ctrl.theme.add(mdl)
        self.__panels = [cls(ctrl, **kwa) for cls in self.KEYS]

        cur = self.__select(self.__initial())
        for panel in self._panels:
            desc = type(panel.plotter).state
            desc.setdefault(panel.plotter, (PlotState.active if panel is cur else
                                            PlotState.disabled))

    @property
    def _panels(self):
        vals = {self.__key(i): i for i in self.__panels}
        return [vals[i] for i in self.__theme.titles]

    def __initial(self):
        "return the initial tab"
        return next(i for i, j in self.KEYS.items() if j == self.__theme.initial)

    @classmethod
    def __key(cls, panel):
        return cls.KEYS[type(panel)]

    @staticmethod
    def __state(panel, val = None):
        if val is not None:
            panel.plotter.state = PlotState(val)
        return panel.plotter.state

    def __select(self, tpe):
        return next(i for i in self._panels if isinstance(i, tpe))

    def ismain(self, ctrl):
        "Allows setting-up stuff only when the view is the main one"
        if "advanced" in ctrl.display.model("keystroke", True):
            def _advanced():
                for panel in self._panels:
                    if self.__state(panel) is PlotState.active:
                        getattr(panel, 'advanced', lambda:None)()
                        break
            ctrl.display.updatedefaults('keystroke', advanced = _advanced)

    def __setstates(self):
        cur = self.__select(self.__initial())
        ind = next((i for i, j in enumerate(self._panels) if j is cur), 0)
        for panel in self._panels[:ind]:
            self.__state(panel, PlotState.disabled)
        self.__state(self._panels[ind], PlotState.active)
        for panel in self._panels[ind+1:]:
            self.__state(panel, PlotState.disabled)
        return ind

    def __createtabs(self, ind):
        panels = [Panel(title = self.__theme.titles[self.__key(i)],
                        child = Spacer())
                  for i in self._panels]
        return Tabs(tabs   = panels,
                    active = ind,
                    name   = self.NAME,
                    width  = self.__theme.width,
                    height = self.__theme.height)

    @staticmethod
    def _addtodoc_oneshot() -> Tuple[str, str]:
        return "display", "applicationstarted"

    def addtodoc(self, ctrl, doc):
        "returns object root"
        super().addtodoc(ctrl, doc)
        tabs  = self.__createtabs(self.__setstates())

        roots = [None]*len(self._panels)
        def _root(ind):
            if roots[ind] is None:
                ret = self._panels[ind].addtodoc(ctrl, doc)
                while isinstance(ret, (tuple, list)) and len(ret) == 1:
                    ret = ret[0]
                if isinstance(ret, (tuple, list)):
                    ret  = layouts.column(ret, **self.defaultsizingmode())

                doc.add_next_tick_callback(lambda: self._panels[ind].plotter.reset(True))
                roots[ind] = ret
            return roots[ind]

        @ctrl.action
        def _py_cb(attr, old, new):
            self._panels[old].activate(False)
            self._panels[new].activate(True)
            ctrl.undos.handle('undoaction',
                              ctrl.emitpolicy.outastuple,
                              (lambda: setattr(tabs, 'active', old),))
            if roots[old] is not None:
                doc.remove_root(roots[old])
                doc.add_root(_root(new))

        tabs.on_change('active', _py_cb)

        def _fcn(**_):
            itm = next(i for i, j in enumerate(self._panels) if j.plotter.isactive())
            doc.add_root(_root(itm))

        one, name = self._addtodoc_oneshot()
        getattr(ctrl, one).oneshot(name, _fcn)

        mode = self.defaultsizingmode()
        return layouts.row(layouts.widgetbox(tabs, **mode), **mode)

    def observe(self, ctrl):
        "observing the controller"
        super().observe(ctrl)
        for panel in self._panels:
            panel.observe(ctrl)

        mdl  = self.__theme
        cur  = ctrl.theme.get(mdl, 'version')
        vers = ctrl.theme.get(mdl, 'version', defaultmodel = True)
        if cur > vers:
            return
        msg = ctrl.theme.get(mdl, 'startup')
        if msg == "":
            msg = mdl.defaultstartup(getattr(ctrl, 'APPNAME', None))
            if msg is None:
                return

        @ctrl.display.observe
        def _onscriptsdone(**_):
            ctrl.theme.update(mdl, version = vers+1)
            ctrl.writeuserconfig()
            dialog(self._doc, body = msg, buttons = "ok")

def initsubclass(name, keys, *_1, **_2):
    "init TabsView subclass"
    def _wrapper(cls):
        cls.KEYS = OrderedDict(keys)
        cls.NAME = name
        return cls
    return _wrapper
