#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"all view aspects here"
from collections         import OrderedDict
from pathlib             import Path
from typing              import Dict, ClassVar, TypeVar, Tuple, Generic, Type, Optional

from bokeh               import layouts
from bokeh.models        import Panel, Spacer, Tabs

from utils.inspection    import templateattribute
from model.plots         import PlotState
from modaldialog         import dialog
from modaldialog.builder import changelog
from view.base           import BokehView
from version             import timestamp as _timestamp

class TabsTheme:  # pylint: disable=too-many-instance-attributes
    "Tabs Theme"
    CHANGELOG = "CHANGELOG.html"

    def __init__(self, initial: str, panels: Dict[Type, str], version = 0, startup = "") -> None:
        self.name:    str            = "app.tabs"
        self.startup: str            = startup
        self.initial: str            = initial
        self.width:   int            = 1100
        self.height:  int            = 30
        self.panelheight: int        = 800
        self.docurl:  str            = "doc"
        self.titles:  Dict[str, str] = OrderedDict()
        for i, j in panels.items():
            self.titles[j] = getattr(i, 'PANEL_NAME', j.capitalize())
        assert self.initial in self.titles
        self.version: int = _timestamp() if version == 0 else version

    def defaultstartup(self, name: str) -> Optional[str]:
        "extracts default startup message from a changelog"
        path = Path(".").absolute()
        for _ in range(4):
            if (path/self.CHANGELOG).exists():
                break
            path = path.parent
        else:
            return None

        path /= self.CHANGELOG
        with open(path, "r", encoding="utf-8") as stream:
            return changelog(stream, name, self.docurl)
        return None


TThemeType = TypeVar("TThemeType", bound = TabsTheme)


class TabsView(Generic[TThemeType], BokehView):
    "A view with all plots"
    KEYS:    ClassVar[Dict[type, str]]
    NAME:    ClassVar[str]
    _tabs:   Tabs
    __theme: TabsTheme

    def __init__(self, ctrl = None, **kwa):
        "Sets up the controller"
        super().__init__(ctrl = ctrl, **kwa)
        mdl             = templateattribute(self, 0)()  # type: ignore
        mdl.width       = kwa.get('width',  ctrl.theme.get("theme", 'appsize')[0])
        mdl.panelheight = (
            kwa.get('height', ctrl.theme.get("theme", 'tabheight'))
            - mdl.height
        )
        self.__theme    = ctrl.theme.add(mdl)
        ctrl.theme.updatedefaults("theme", tabheight = mdl.panelheight)
        self.__panels   = [cls(ctrl) for cls in self.KEYS]

        cur = self.__select(self.__initial())
        for panel in self._panels:
            desc = type(panel.plotter).state
            desc.setdefault(panel.plotter, (PlotState.active if panel is cur else
                                            PlotState.disabled))

    def swapmodels(self, ctrl):
        "swap models"
        self.__theme = ctrl.theme.swapmodels(self.__theme)
        for i in self.__panels:
            if hasattr(i, 'swapmodels'):
                i.swapmodels(ctrl)

    @property
    def _panels(self):
        vals = {self.__key(i): i for i in self.__panels}
        return [vals[i] for i in self.__theme.titles]

    @property
    def current(self):
        "return the current plotter"
        return next(j.plotter for j in self._panels if j.plotter.isactive())

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
        panels = [
            Panel(
                title = self.__theme.titles[self.__key(i)],
                child = Spacer(width  = self.__theme.width, height = 0)
            )
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

        mode = self.defaultsizingmode(width = self.__theme.width, height = self.__theme.height)
        row  = layouts.column(
            layouts.widgetbox(tabs, **mode),
            Spacer(width = self.__theme.width, height = 0),
            **mode
        )

        @ctrl.action
        def _py_cb(attr, old, new):
            self._panels[old].activate(False)
            self._panels[new].activate(True)
            ctrl.undos.handle('undoaction',
                              ctrl.emitpolicy.outastuple,
                              (lambda: setattr(tabs, 'active', old),))
            if roots[old] is not None:
                children = list(row.children)[:-1]
                row.update(children = children+[_root(new)])

        tabs.on_change('active', _py_cb)

        def _fcn(**_):
            itm      = next(i for i, j in enumerate(self._panels) if j.plotter.isactive())
            children = list(row.children)[:-1]
            row.update(children = children+[_root(itm)])

        one, name = self._addtodoc_oneshot()
        getattr(ctrl, one).oneshot(name, _fcn)
        return row

    def observe(self, ctrl):
        "observing the controller"
        super().observe(ctrl)
        for panel in self._panels:
            panel.observe(ctrl)

        @ctrl.display.observe
        def _onchangelog(**_):
            doc = self.__theme.defaultstartup(getattr(ctrl, 'APPNAME', None))
            if doc:
                dialog(self._doc, body = doc, buttons = "ok")

        cur  = ctrl.theme.get(self.__theme, 'version')
        vers = ctrl.theme.get(self.__theme, 'version', defaultmodel = True)
        if None not in (cur, vers) and cur <= vers:
            @ctrl.display.observe
            def _onscriptsdone(**_):
                mdl = self.__theme
                ctrl.theme.update(mdl, version = vers+1)
                ctrl.writeuserconfig()

                msg = ctrl.theme.get(mdl, 'startup')
                if msg == "":
                    msg = mdl.defaultstartup(getattr(ctrl, 'APPNAME', None))
                if msg:
                    dialog(self._doc, body = msg, buttons = "ok")

def initsubclass(name, keys, *_1, **_2):
    "init TabsView subclass"
    def _wrapper(cls):
        cls.KEYS = OrderedDict(keys)
        cls.NAME = name
        return cls
    return _wrapper
