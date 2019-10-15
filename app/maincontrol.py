#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"The main controller"
import sys
from   contextlib              import contextmanager
from   functools               import wraps
from   copy                    import deepcopy
from   typing                  import Dict, Any, List, Union, Iterable

import bokeh.models as _models
from   bokeh.themes            import Theme
from   bokeh.layouts           import layout, widgetbox
from   bokeh.models            import Paragraph
from   bokeh.document          import Document

from   control.event           import EmitPolicy
from   control.decentralized   import DecentralizedController
from   control.action          import ActionDescriptor
from   undo.control            import UndoController
from   view.keypress           import DpxKeyEvent
from   model.maintheme         import MainTheme, AppTheme
from   utils.logconfig         import getLogger
from   .configuration          import ConfigurationIO
from   .scripting              import orders
LOGS = getLogger(__name__)

class WithMessage:
    "wraps a method to execute it safely"
    def __init__(self, ctrl, exceptions):
        self.ctrl: 'BaseSuperController' = ctrl
        self.warns: List[type] = (
            [exceptions]     if isinstance(exceptions, type) else
            list(exceptions) if exceptions is not None       else
            []
        )

    def __enter__(self) -> None:
        return None

    def __exit__(self, tpe, val, bkt) -> bool:
        if val is not None:
            self.__apply(val)
        return True

    def __call__(self, fcn):
        @wraps(fcn)
        def _wrapped(*args, **kwa):
            try:
                return fcn(*args, **kwa)
            except Exception as exc:  # pylint: disable=broad-except
                self.__apply(exc)
                raise
        return _wrapped

    def __apply(self, val):
        LOGS.exception(val)
        msg = (
            val,
            'warning' if self.warns and isinstance(val, self.warns) else 'error'
        )
        self.ctrl.display.update('message', message = msg)

class DisplayController(DecentralizedController):
    "All temporary information related to one application run"
    CATCHERROR = True

    def __repr__(self):
        return "DisplayControl"

class ThemeController(DecentralizedController):
    "All static information that should remain from application run to application run"
    def __repr__(self):
        return "ThemeControl"

    def initializetheme(self, doc:Document):
        "init the theme"
        theme = self.model("main").theme
        if theme:
            doc.theme = Theme(json = theme)

        @self.observe
        def _onmain(old = None, ** _):
            if len({'themename', 'customlight', 'customdark'} & set(old)):
                theme     = self.model("main").theme
                doc.theme = Theme(json = theme)

    def updatetheme(self, doc:Document, **values):
        "change the theme"
        theme  = doc.theme
        cur    = deepcopy(dict(
            getattr(theme, '_json')['attrs'],
            **getattr(theme, '_by_class_cache')
        ))
        isdiff = False
        for i, j in values.items():
            if isinstance(j, dict):
                tmp = dict(cur[i], **j)
                if tmp and tmp != j:
                    cur[i] = tmp
                    isdiff = True

            elif i.startswith("font"):
                for name, attrs in cur.items():
                    lst = dict((k, j) for k in attrs if k.endswith(i))
                    tmp = dict(attrs, **lst)
                    if tmp and tmp != attrs:
                        cur[name] = tmp
                        isdiff    = True

            else:
                for name, attrs in cur.items():
                    if hasattr(getattr(_models, name, None), i):
                        tmp = dict(attrs, **{i: j})
                        if 'font_size' in i and j is None:
                            tmp.pop(i)
                        if tmp and tmp != attrs:
                            cur[name] = tmp
                            isdiff    = True

        if isdiff:
            cur = {
                i: j
                for i,j in cur.items() if not isinstance(j, dict) or j
            }
            name = 'custom' + ('dark' if 'dark' in self.model("main").themename else 'light')
            self.update("main", **{name: {"attrs": cur}, 'themename': name})

    @staticmethod
    def gettheme(doc:Document, attr: str):
        "return the current value"
        for i in getattr(doc.theme, '_by_class_cache').values():
            if len(i) == 0:
                continue

            val = (i.get(attr, None)        if not attr.startswith('font') else
                   next((k for j, k in i.items() if j.endswith(attr)), None))

            if val is not None:
                return val
        return None

class BaseSuperController:
    """
    Main controller: contains all sub-controllers.
    These share a common dictionnary of handlers
    """
    APPNAME     = 'Track Analysis'
    APPSIZE     = list(AppTheme().appsize)
    FLEXXAPP    = None
    action      = ActionDescriptor()
    computation = ActionDescriptor()

    def __init__(self, view, **kwa):
        self.topview = view
        self.undos   = UndoController(**kwa)
        self.theme   = ThemeController()
        self.theme.add(MainTheme())
        self.display = DisplayController()

        for i, j in self.__dict__.items():
            if i != 'theme' and callable(getattr(j, 'linkdepths', None)):
                j.linkdepths(self.theme)

        self._config_counts = [False]

    emitpolicy = EmitPolicy

    def __undos__(self, wrapper):
        for i in self.__dict__.values():
            getattr(i, '__undos__', lambda _: None)(wrapper)

    @classmethod
    def open(cls, viewcls, doc, **kwa):
        "starts the application"
        # pylint: disable=protected-access
        return cls(None)._open(viewcls, doc, kwa)

    def close(self):
        "remove controller"
        top, self.topview = self.topview, None
        if top is None:
            return

        self.writeuserconfig()
        for i in tuple(self.__dict__.values()) + (top, self.FLEXXAPP):
            getattr(i, 'close', lambda: None)()

    def writeuserconfig(self, name = None, saveall = False, index = 0, **kwa):
        "writes the config"
        ConfigurationIO(self).writeuserconfig(self._getmaps(), name, saveall, index = index, **kwa)

    @classmethod
    def launchkwargs(cls, **kwa) -> Dict[str, Any]:
        "updates kwargs used for launching the application"
        maps = cls.__apptheme()
        DisplayController.CATCHERROR = maps['config']['catcherror']
        kwa.setdefault("title",  maps['theme']["appname"])
        kwa.setdefault("size",   maps['theme']['appsize'])
        return kwa

    def withmessage(
            self,
            fcn = None,
            exceptions: Union[None, type, Iterable[type]] = None
    ):
        "wraps a method to execute it safely"
        ret = WithMessage(self, exceptions)
        return ret if fcn is None else ret(fcn)

    def _open(self, viewcls, doc, kwa):
        @contextmanager
        def _test(msg):
            try:
                yield
            except Exception as exc:  # pylint: disable=broad-except
                LOGS.critical(msg)
                LOGS.exception(msg)
                if doc:
                    doc.add_root(widgetbox(Paragraph(text = f"[{type(exc)}] {exc}")))
                raise

        keys = None
        try:
            with _test('Could not read main theme'):
                self.theme.add(AppTheme(**self.__apptheme()['theme']))

            with _test("Could not create GUI instance"):
                keys         = DpxKeyEvent(self)
                self.topview = viewcls(self, **kwa)
                for i in self.topview.views:
                    getattr(i, 'swapmodels', lambda *_: None)(self)
                if len(self.topview.views) and hasattr(self.topview.views[0], 'ismain'):
                    self.topview.views[0].ismain(self)

            with _test("Could not configure GUI"):
                self._configio()

            if doc is not None:
                with _test("Could not setup observers"):
                    self._observe(keys)
                with _test("Could not create GUI"):
                    self._bokeh(keys, doc)
                with _test("Could not handle applicationstarted event"):
                    self.display.handle('applicationstarted', self.display.emitpolicy.nothing)
        except Exception:  # pylint: disable=broad-except
            pass
        return self

    def _observe(self, keys):
        "Returns the methods for observing user start & stop action delimiters"
        if keys:
            keys.observe(self)
        for i in self.topview.views:
            getattr(i, 'observe', lambda *_: None)(self)

        # now observe all events that should be saved in the config
        self._config_counts = [False]

        @self.display.observe
        def _onstartaction(recursive = None, **_):
            if recursive is False:
                self._config_counts[0]  = False

        def _onconfig(*_1, **_2):
            self._config_counts[0] = True

        args = self._observeargs()
        assert len(args) % 2 == 0
        for i in range(0, len(args), 2):
            args[i].observe(args[i+1], _onconfig)

        for i in self.theme.values():
            self.theme.observe(i, _onconfig)

        @self.display.observe
        def _onstopaction(recursive = None, **_):
            if recursive is False:
                self._config_counts, val = [False], self._config_counts[0]
                if val:
                    self.writeuserconfig()

    def _bokeh(self, keys, doc):
        for mdl in orders().dynloads():
            getattr(sys.modules.get(mdl, None), 'document', lambda x: None)(doc)

        #  retry saving the cache as new bokeh models may have been added
        #  since app.launcher's call
        from .launcher import CAN_LOAD_JS
        if CAN_LOAD_JS:
            from utils.gui import storedjavascript
            storedjavascript(CAN_LOAD_JS, self.APPNAME)

        roots = [
            getattr(i, 'addtodoc', lambda *_: None)(self, doc)
            for i in self.topview.views
        ]
        roots = [i for i in roots if i is not None]

        if not roots:
            return

        self.theme.initializetheme(doc)

        while isinstance(roots, (tuple, list)) and len(roots) == 1:
            roots = roots[0]

        if not isinstance(roots, (tuple, list)):
            roots = (roots,)

        keys.addtodoc(self, doc)
        if isinstance(roots, (tuple, list)) and len(roots) == 1:
            doc.add_root(roots[0])
        else:
            mode = self.theme.get('main', 'sizingmode')
            doc.add_root(layout(roots, sizing_mode = mode))

    def _configio(self):
        cnf  = ConfigurationIO(self)
        maps = self._getmaps()

        # 1. write the whole config down
        cnf.writeuserconfig(maps, "defaults",   True,  index = 1)

        # 2. read & write the user-provided config: discards unknown keys
        cnf.writeuserconfig(cnf.readconfig(maps, "userconfig"), "userconfig", False, index = 0)

        # read the config from files:
        self._setmaps(cnf.readuserconfig(maps))

        orders().config(self)

    def _getmaps(self):
        maps = {'theme':  {'appsize': self.APPSIZE, 'appname': self.APPNAME},
                'config': {'catcherror': DisplayController.CATCHERROR}}
        keys = {i for i in self.theme.current.keys()
                if type(self.theme.model(i)).__name__.endswith("Config")}
        outs = {f'{"config." if i in keys else "theme."}{i}': j
                for i, j in self.theme.config.items()}
        for i, j in maps.items():
            j.update(outs.pop(i, {}))
        maps.update(outs)
        return maps

    def _setmaps(self, maps):
        for i, j in maps.items():
            if i == 'theme':
                self.theme.update(i, **j)
            if i.startswith('theme.') and i[6:] in self.theme and j:
                self.theme.update(i[6:], **j)
            elif i.startswith('config.') and i[7:] in self.theme and j:
                self.theme.update(i[7:], **j)

    @classmethod
    def __apptheme(cls) -> Dict[str, Any]:
        "updates kwargs used for launching the application"
        cnf   = ConfigurationIO(cls)
        maps  = {'theme': {'appsize': cnf.appsize, 'appname': cnf.appname},
                 'config': {'catcherror': DisplayController.CATCHERROR}}
        maps = cnf.readuserconfig(maps, update = True)
        return maps

    def _observeargs(self):
        raise NotImplementedError()

def createview(cls, main, controls, views):
    "Creates a main view"
    controls = (cls,)+tuple(controls)
    views    = (main,)+tuple(views)
    return ConfigurationIO.createview((cls,)+controls, views)
