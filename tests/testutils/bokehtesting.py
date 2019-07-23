#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Utils for testing views"
from   importlib import import_module
from   pathlib   import Path
from   time      import time as process_time
from   typing    import Optional, Union, Sequence, Any, cast
from   threading import Thread
from   functools import partial
import os
import tempfile
import warnings
import inspect
import logging
import webbrowser

warnings.filterwarnings(
    'ignore',
    category = DeprecationWarning,
    message  = ".*Using or importing the ABCs from 'collections'.*"
)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category = DeprecationWarning)
    # pylint: disable=unused-import
    import pytest
    from bokeh.model                import Model
    from bokeh.document             import Document
    from bokeh.server.server        import Server
    import bokeh.core.properties    as     props
    import bokeh.plotting.figure

# pylint: disable=wrong-import-position
from tornado.platform.asyncio       import AsyncIOMainLoop
import webruntime

import app.configuration as _conf
from utils.logconfig                import getLogger, iterloggers
from view.static                    import ROUTE
from view.keypress                  import DpxKeyEvent

LOGS = getLogger(__name__)

class ErrorHandler(logging.Handler):
    """
    A handler class which  deals with errors comming from the server
    """
    def __init__(self):
        super().__init__(level = logging.ERROR)
        self.lst: list = []

    def emit(self, record):
        "emit"
        self.lst.append(record)

class DpxTestLoaded(Model):
    """
    This starts tests once flexx/browser window has finished loading
    """
    __javascript__     = ROUTE+"/jquery.min.js"
    __implementation__ = 'bokehtesting.ts'
    done        = props.Int(0)
    event       = props.Dict(props.String, props.Any)
    event_cnt   = props.Int(0)
    modelid     = props.String('')
    attrs       = props.List(props.String, default = [])
    attr        = props.String()
    value       = props.Any()
    value_cnt   = props.Int(0)
    debug       = props.String()
    warn        = props.String()
    info        = props.String()
    def __init__(self, **kwa):
        super().__init__(**kwa)
        self.on_change("debug", self.__log_cb)
        self.on_change("warn",  self.__log_cb)
        self.on_change("info",  self.__log_cb)

    @staticmethod
    def __log_cb(attr, old, new):
        if new != '':
            getattr(LOGS, attr)('JS <- '+new)

    def press(self, key, model):
        "Sets-up a new keyevent in JS"
        assert model is None or key in model
        val = '-' if key == '-' else key.split('-')[-1]
        evt = dict(alt   = 'Alt-'     in key,
                   shift = 'Shift-'   in key,
                   ctrl  = 'Control-' in key,
                   meta  = 'Meta-'    in key,
                   key   = val)
        self.modelid = model.id
        self.event = evt
        LOGS.debug("pressing: %s", key)
        self.event_cnt += 1

    def change(self, model:Model, attrs: Union[str, Sequence[str]], value: Any):
        "Changes a model attribute on the browser side"
        self.modelid = model.id
        self.attrs = list(attrs)[:-1] if isinstance(attrs, (tuple, list)) else []
        self.attr  = attrs[-1]        if isinstance(attrs, (tuple, list)) else attrs
        self.value = value
        LOGS.debug("changing: %s = %s", attrs, value)
        self.value_cnt += 1

class WidgetAccess:
    "Access to bokeh models"
    _none = type('_none', (), {})
    def __init__(self, docs, key = None):
        self._docs = docs if isinstance(docs, (list, tuple)) else (docs,)
        self._key  = key

    def get(self, value):
        "finds the first item"
        if isinstance(value, Model):
            return value
        value = ({'name': value} if isinstance(value, str) else
                 {'type': value} if isinstance(value, type) else
                 dict(value))
        return next(iter(self._docs[0].select(value)))

    def __getitem__(self, value):
        if isinstance(value, type):
            if self._key is not None:
                val = next((i for doc in self._docs for i in doc.select({'type': value})),
                           self._none)
                if val is not self._none:
                    return val
            return next(i for doc in self._docs for i in doc.select({'type': value}))

        itms: tuple = tuple()
        for doc in self._docs:
            itms += tuple(doc.select({'name': value}))
        if len(itms) > 0:
            return WidgetAccess(itms)
        key = value if self._key is None else self._key + '.' + value
        return WidgetAccess(tuple(self._docs), key)

    def __getattr__(self, key):
        return super().__getattribute__(key) if key[0] == '_' else getattr(self(), key)

    def __setattr__(self, key, value):
        if key[0] == '_':
            super().__setattr__(key, value)
        else:
            setattr(self(), key, value)

    def __call__(self):
        if self._key is not None:
            raise KeyError("Could not find "+ self._key)
        return self._docs[0]

class BaseSeleniumAccess:
    "Access to selenium driver"
    def __init__(self, server):
        self.server = server
        self.driver = server.driver

    def __getitem__(self, name):
        if isinstance(name, list):
            return (self[i] for i in name)

        fcn = f"find_element{'s' if isinstance(name, tuple) and Ellipsis in name else ''}_by_"
        if 's' in fcn:
            name = next(i for i in name if i is not Ellipsis)

        if name.startswith("#"):
            return getattr(self.driver, f"{fcn}id")(name[1:])
        if name.startswith("."):
            return getattr(self.driver, f"{fcn}class_name")(name[1:])
        if name.startswith("/"):
            return getattr(self.driver, f"{fcn}xpath")(name)
        raise NotImplementedError()

    def __setitem__(self, name, value):
        "click on a button"
        elem = self[name]
        self.driver.execute_script(f"arguments[0].setAttribute('value', '{value}')", elem)

    def select(self, name, value):
        "select a value in a dropdown"
        # pylint: disable=import-error
        from selenium.webdriver.support.ui import Select
        select = Select(self[name])
        select.select_by_value(value)

    def enabled(self, name):
        "click on a button"
        return not self[name].is_enabled()

    def classnames(self, name):
        "click on a button"
        return self[name].get_attribute("class")

    def click(self, name, wait = True):
        "click on a button"
        self[name].click()
        if wait:
            self.server.wait()

class ModalAccess(BaseSeleniumAccess):
    "Access to selenium driver"
    def __init__(self, server, btn: str, done = False):
        super().__init__(server)
        self.cancel = not done
        self.btn    = btn

    def close(self, done = None):
        "click on the modal's apply button"
        if done is None:
            done = not self.cancel

        super().click(f".dpx-modal-{'done' if done else 'cancel'}")

    def isdisplayed(self) -> bool:
        "checks whether it's open"
        from selenium.common.exceptions import NoSuchElementException
        try:
            return self[".dpx-modal-done"].is_displayed()
        except NoSuchElementException:
            return False

    def open(self, btn = None):
        "starts the modal dialog"
        # pylint: disable=import-error
        from selenium.common.exceptions import ElementClickInterceptedException
        for _ in range(3):
            try:
                super().click(btn if btn else self.btn)
            except ElementClickInterceptedException:
                elem = self.driver[".bbm-wrapper"]
                if elem:
                    self.driver.driver.execute_script(
                        f"arguments[0].style.visibility='hidden'",
                        elem
                    )
            else:
                break

    def toggle(self, text) -> 'ModalAccess':
        "sets the input"
        self.input(text).click()
        return self

    def input(self, text, value = Ellipsis):
        "sets the input"
        txt = f'//td[text()="{text}"]/following::input[1]'
        if value is Ellipsis:
            return self[txt]

        self[txt] = value
        return self

    def tab(self, text) -> 'ModalAccess':
        "selects a tab"
        super().click(f'//button[text()="{text}"]', False)
        return self

    def click( # pylint: disable=arguments-differ
            self,
            btn: Optional[str] = None,
            done: Optional[bool] = None
    ):
        "opens and closes the dialog"
        self.open(btn)
        self.close(done)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

class SeleniumAccess(BaseSeleniumAccess):
    "Access to selenium driver"
    def modal(self, btn: str, done: bool = True) -> ModalAccess:
        "return a ModalAccess"
        return ModalAccess(self.server, btn, done)

class _ManagedServerLoop: # pylint: disable=too-many-instance-attributes
    """
    lets us use a current IOLoop with "with"
    and ensures the server unlistens
    """
    loop     = property(lambda self: self.server.io_loop)
    ctrl     = property(lambda self: getattr(self.view, '_ctrl'))
    @property
    def loading(self) -> Optional[DpxTestLoaded]:
        "returns the model which allows tests to javascript"
        return next(
            (i for i in getattr(self.doc, 'roots', []) if isinstance(i, DpxTestLoaded)),
            None
        )

    class _Dummy:
        @staticmethod
        def setattr(*args):
            "dummy"
            return setattr(*args)

    __warnings: Any
    __hdl:      ErrorHandler
    def __init__(self, mkpatch, kwa:dict, filters) -> None:
        self.server: Server   = None
        self.driver: Any      = None
        self.view:   Any      = None
        self.doc:    Document = None
        self.monkeypatch      = self._Dummy() if mkpatch is None else mkpatch # type: ignore
        self.kwa              = kwa
        self.headless         = (
            os.environ.get("DPX_TEST_HEADLESS", '').lower().strip() in ('true', '1', 'yes')
            or 'DISPLAY' not in os.environ
        )
        self.headless         = kwa.pop('headless', self.headless)
        self.filters: list    = [] if filters is None else filters
        self.filters.extend((
            ('ignore', '.*inspect.getargspec().*'),
            (RuntimeWarning, ".*coroutine 'HTTPServer.close_all_connections'.*"),
            (DeprecationWarning, '.*elementwise == comparison failed.*'),
        ))

    @staticmethod
    def __import(amod):
        if not isinstance(amod, str):
            return amod

        if '.' in amod and 'A' <= amod[amod.rfind('.')+1] <= 'Z':
            modname     = amod[:amod.rfind('.')]
            attr:tuple  = (amod[amod.rfind('.')+1:],)
        else:
            modname = amod
            attr    = tuple()

        mod = __import__(modname)
        for i in tuple(modname.split('.')[1:]) + attr:
            mod = getattr(mod, i)
        return mod

    def __patchserver(self, server):
        def _open(_, viewcls, doc, _func_ = server.MainView.MainControl.open, **kwa):
            doc.add_root(DpxTestLoaded())
            self.doc  = doc
            ctrl      = server.MainView.MainControl(None)
            self.view = getattr(ctrl, '_open')(viewcls, doc, kwa).topview
            setattr(self.view, '_ctrl', ctrl)
            return ctrl
        server.MainView.MainControl.open = classmethod(_open)

        def _close(this, *_1, _func_ = server.MainView.close, **_):
            self.server = None
            ret = _func_(this)
            return ret

        server.MainView.close = _close

    def __getlauncher(self):
        tmpapp, mod, fcn = self.kwa.pop('_args_')
        app              = self.__import(tmpapp)
        if not isinstance(app, type):
            from view.base import BokehView
            pred = lambda i: (isinstance(i, type)
                              and i.__module__.startswith(app.__name__)
                              and issubclass(i, BokehView))
            pot  = tuple(i for _, i in inspect.getmembers(app, pred))
            assert len(pot) == 1
            app  = pot[0]

        return app, getattr(self.__import(mod), fcn)

    def __buildserver(self, kwa):
        AsyncIOMainLoop().make_current()

        app, launch = self.__getlauncher()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*inspect.getargspec().*')
            import utils.gui as _gui
            from   app.scripting import addload
            from   utils.gui     import storedjavascript

            name = "___"
            for j in inspect.getouterframes(inspect.currentframe()):
                if j.function.startswith("test_"):
                    name = j.function
            addload("view.static", "modaldialog")
            _gui.storedjavascript = lambda *_: storedjavascript("tests", name)
            kwa.setdefault('port', 'random')
            if self.headless and kwa.get('runtime', '') != 'none':
                kwa.pop('runtime', None)
            try:
                server = launch(app, **kwa)
            finally:
                _gui.storedjavascript = storedjavascript

        self.__patchserver(server)
        return server

    def __set_warnings(self):
        self.__warnings = warnings.catch_warnings()
        self.__warnings.__enter__()
        for i in self.filters:
            if isinstance(i, dict):
                warnings.filterwarnings('ignore', **i)
            elif isinstance(i[-1], dict):
                warnings.filterwarnings(*i[:-1], **i[-1])
            elif (
                    isinstance(i[0], type)
                    and issubclass(i[0], Exception)
                    and len(i) == 2
            ):
                warnings.filterwarnings('ignore', category = i[0], message = i[1])
            else:
                warnings.filterwarnings(*i)

    def __set_display(self):
        if not self.headless:
            return

        for i in ('Mozilla', 'Chrome'):
            cls = getattr(webbrowser, i)
            self.monkeypatch.setattr(
                cls,
                'remote_args',
                ['--headless']+cls.remote_args
            )

        old = getattr(webruntime.BaseRuntime, '_start_subprocess')
        def _start_subprocess(self, cmd, shell=False, **env):
            return old(self, cmd+['--headless'], shell, **env)
        self.monkeypatch.setattr(
            webruntime.BaseRuntime,
            '_start_subprocess',
            _start_subprocess
        )

    def __set_handler(self):
        self.__hdl = ErrorHandler()
        getLogger().addHandler(self.__hdl)

    def __start(self):
        haserr = [False]
        thread = [self.driver is not None, None]
        first  = [True]
        def _start(time = process_time()):
            "Waiting for the document to load"
            if getattr(self.loading, 'done', False):
                LOGS.debug("done waiting")
                setattr(self.ctrl, 'CATCHERROR', False)

                root = next((i for i in self.doc.roots if hasattr(i, 'keycontrol')), None)
                if getattr(root, 'keycontrol', False):
                    assert first[0]
                    first[0] = False
                    self.doc.add_next_tick_callback(lambda: setattr(root, 'keycontrol', False))
                    self.loop.call_later(0.5, _start)
                else:
                    self.loop.call_later(2., self.loop.stop)
            elif process_time()-time > 20:
                haserr[0] = True
                self.loop.stop()
            else:
                if thread[0]:
                    fcn       = partial(self.driver.get, f"http://localhost:{self.server.port}")
                    thread[0] = False
                    thread[1] = Thread(target = fcn)
                    thread[1].start()
                LOGS.debug("waiting %s", process_time()-time)
                self.loop.call_later(0.5, _start)
        _start()

        self.server.start()
        self.loop.start()
        assert len(self.__hdl.lst) == 0, "gui construction failed"
        assert not haserr[0], "could not start gui"
        if thread[1] is not None:
            thread[1].join()

    def __enter__(self):
        self.__set_display()
        self.__set_handler()
        self.__set_warnings()
        kwa    = dict(self.kwa)
        if kwa.get('runtime', '') == 'selenium':
            kwa.update(runtime = 'none')
            # pylint: disable=import-error
            from selenium.webdriver import Firefox, FirefoxOptions
            opts          = FirefoxOptions()
            opts.headless = self.headless
            self.driver   = Firefox(options = opts)

        self.server = self.__buildserver(kwa)
        self.__start()
        return self

    def __exit__(self, *_):
        if self.server is not None:
            self.wait()
            self.quit()
        self.__warnings.__exit__(*_)
        if self.driver is not None:
            self.driver.quit()
            self.driver = None

        for _1, j  in iterloggers():
            j.removeHandler(self.__hdl)

        assert len(self.__hdl.lst) == 0, str(self.__hdl.lst)

    PATH = None
    @classmethod
    def path(cls, path: Union[Sequence[str], str]) -> Union[str, Sequence[str]]:
        "returns the path to testing data"
        pathfcn = cls.PATH
        if pathfcn is not None:
            LOGS.debug("Test is opening: %s", path)
            return pathfcn(path) # pylint: disable=not-callable
        raise NotImplementedError()

    def cmd(self, fcn, *args, andstop = True, andwaiting = 2., rendered = False, **kwargs):
        "send command to the view"
        if rendered:
            self.ctrl.display.oneshot(
                rendered if isinstance(rendered, str) else "rendered",
                lambda *_1, **_2: self.wait()
            )
            andstop = False
        if andstop:
            def _cmd():
                LOGS.debug("running: %s(*%s, **%s)", fcn.__name__, args, kwargs)
                fcn(*args, **kwargs)
                LOGS.debug("done running and waiting %s", andwaiting)
                self.loop.call_later(andwaiting, self.loop.stop)
        else:
            def _cmd():
                LOGS.debug("running: %s(*%s, **%s)", fcn.__name__, args, kwargs)
                fcn(*args, **kwargs)
                LOGS.debug("done running and not stopping")
        self.doc.add_next_tick_callback(_cmd)
        if not self.loop.asyncio_loop.is_running():
            self.loop.start()

    def wait(self, time = 2., rendered = False):
        "wait some more"
        if rendered:
            self.cmd(lambda: None, andwaiting = time, rendered = True)
        else:
            self.cmd(lambda: None, andwaiting = time)

    def quit(self):
        "close the view"
        def _quit():
            server = self.server
            server.unlisten()
            self.ctrl.close()
            server.stop()

        self.cmd(_quit, andstop = False)

    def save(self, path: str, andpress = True, waitfor = 10, **kwa):
        "save to a path"
        import view.dialog  # pylint: disable=import-error
        def _tkopen(*_1, **_2):
            LOGS.info("saving to path '%s'", path)
            return path

        self.monkeypatch.setattr(view.dialog, '_tksave', _tkopen)
        self.monkeypatch.setattr(view.dialog.BaseFileDialog, '_HAS_ZENITY', False)
        self.monkeypatch.setattr(view.dialog.BaseFileDialog, '_calltk', _tkopen)
        if andpress:
            if waitfor and Path(path).exists():
                Path(path).unlink()
            self.press('Control-s',rendered = False, **kwa)
            for _ in range(waitfor if isinstance(waitfor, int) else 10):
                if Path(path).exists() and (not callable(waitfor) or waitfor()):
                    break
                self.wait()
            else:
                assert False

    def load(self, path: Union[Sequence[str], str], andpress = True, rendered = True, **kwa):
        "loads a path"
        import view.dialog  # pylint: disable=import-error
        def _tkopen(*_1, **_2):
            out = self.path(path)
            LOGS.debug("opening path %s -> %s", path, out)
            return out

        self.monkeypatch.setattr(view.dialog, '_tkopen', _tkopen)
        self.monkeypatch.setattr(view.dialog.BaseFileDialog, '_HAS_ZENITY', False)
        self.monkeypatch.setattr(view.dialog.BaseFileDialog, '_calltk', _tkopen)
        if andpress:
            self.press('Control-o',rendered = rendered, **kwa)

    def get(self, clsname, attr):
        "Returns a private attribute in the view"
        key = '_'+clsname+'__'+attr
        if key in self.view.__dict__:
            return self.view.__dict__[key]

        key = '_'+attr
        if key in self.view.__dict__:
            return self.view.__dict__[key]

        return self.view.__dict__[attr]

    def press(self, key:str, src = None, **kwa):
        "press one key in python server"
        loading = cast(DpxTestLoaded, self.loading)
        if src is None:
            for root in self.doc.roots:
                if isinstance(root, DpxKeyEvent):
                    self.cmd(loading.press, key, root, **kwa)
                    break
            else:
                raise KeyError("Missing DpxKeyEvent in doc.roots")
        else:
            self.cmd(loading.press, key, src, **kwa)

    def click(self, model: Union[str,dict,Model], **kwa):
        "Clicks on a button on the browser side"
        mdl = self.widget.get(model)
        self.change(mdl, 'clicks', mdl.clicks+1, **kwa)

    def change(self,        # pylint: disable=too-many-arguments
               model: Union[str,dict,Model],
               attrs: Union[str, Sequence[str]],
               value: Any,
               browser     = True,
               withpath    = None,
               withnewpath = None,
               rendered    = False):
        "Changes a model attribute on the browser side"
        mdl = self.widget.get(model)
        if withnewpath is not None or withpath is not None:
            import view.dialog  # pylint: disable=import-error
            if withnewpath is not None:
                def _tkopen1(*_1, **_2):
                    return withnewpath
                fcn = _tkopen1
            else:
                def _tkopen2(*_1, **_2):
                    return self.path(withpath)
                fcn = _tkopen2
            self.monkeypatch.setattr(view.dialog, '_tkopen', fcn)
            self.monkeypatch.setattr(view.dialog.BaseFileDialog, '_HAS_ZENITY', False)
            self.monkeypatch.setattr(view.dialog.BaseFileDialog, '_calltk', fcn)

        if browser:
            self.cmd(cast(DpxTestLoaded, self.loading).change, mdl, attrs, value,
                     rendered = rendered)
            return

        if rendered:
            self.ctrl.display.oneshot("rendered", lambda *_1, **_2: self.wait())
        assert isinstance(attrs, str)
        @self.cmd
        def _cb():
            setattr(mdl, cast(str, attrs), value)

    @property
    def widget(self):
        "Returns something to access web elements"
        return WidgetAccess(self.doc)

    @property
    def selenium(self):
        "Returns something to access web elements"
        assert self.driver is not None
        return SeleniumAccess(self)

    STORE = None
    @property
    def savedconfig(self):
        "return the saved config"
        if self.STORE is None:
            raise NotImplementedError()
        taskstore = import_module(self.STORE)
        path      = (
            _conf.ConfigurationIO(self.ctrl)
            .configpath(next(taskstore.iterversions('config')))
        )
        return taskstore.load(path)

class BokehAction:
    "All things to make gui testing easy"
    def __init__(self, mkpatch):
        if not mkpatch:
            from . import getmonkey
            mkpatch = getmonkey()

        self.monkeypatch = mkpatch
        tmp = tempfile.mktemp()+"_test"
        class _Dummy:
            user_config_dir = lambda *_: tmp+"/"+_[-1]
        self.monkeypatch.setattr(_conf, 'appdirs', _Dummy)
        self.server: _ManagedServerLoop = None

    def start( # pylint: disable=too-many-arguments
            self,
            app:Union[type, str],
            mod:     str  = 'default',
            filters: list = None,
            launch: bool  = True,
            keep:   bool  = True,
            **kwa
    ) -> _ManagedServerLoop:
        "Returns a server managing context"
        kwa['_args_'] = app, mod, 'launch' if launch else 'serve'
        server        = _ManagedServerLoop(self.monkeypatch, kwa, filters)
        if keep:
            self.server = server
            self.server.__enter__()
        return server

    def __enter__(self):
        return self

    def __exit__(self, *_):
        "stop server"
        act, self.server = self.server, None
        return act.__exit__(*_) if act is not None else None

    def setattr(self, *args, **kwargs):
        "apply monkey patch"
        self.monkeypatch.setattr(*args, **kwargs)
        return self
