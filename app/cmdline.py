#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Runs an app"
from   copy      import copy, deepcopy
from   pathlib   import Path
from   time      import time
from   functools import partial
import logging
import inspect
import warnings

import numpy as np
import click

import utils.warningsconfig   #  pylint: disable=unused-import
from utils.logconfig    import getLogger
from bokeh.resources    import DEFAULT_SERVER_PORT # pylint: disable=wrong-import-order
from app.scripting      import INITIAL_ORDERS
LOGS = getLogger()

def _without_cls(vname, cname):
    viewmod = __import__(vname)
    for name in vname.split('.')[1:]:
        viewmod = getattr(viewmod, name)

    pred = lambda i: (isinstance(i, type)
                      and (i.__module__ == viewmod.__name__
                           or i.__module__.startswith(viewmod.__name__+'.'))
                      and i.__name__.lower() == cname)
    pot  = tuple(i for _, i in inspect.getmembers(viewmod, pred))
    if len(pot) == 0:
        return _without_cls(vname+".view", cname)
    assert len(pot) == 1
    return pot[0]

def _from_path(view):
    pview = Path(view)
    if pview.exists():
        name = (pview.parent.stem+'view' if pview.stem == 'view'        else
                pview.stem               if pview.stem.endswith('view') else
                pview.stem+'view')

        mod = str(pview.parent/pview.stem).replace('/', '.').replace('\\', '.')
        return _without_cls(mod, name)
    return None

def _from_module(view):
    if '/' in view or '\\' in view:
        view = view.replace('/', '.').replace('\\', '.')
        if view.endswith('.py'):
            view = view[:-3]

    if view.startswith('view.'):
        view = view[5:]

    if '.' not in view:
        view = view.lower()+'.'+view
    try:
        ind  = view.rfind('.')
        name = view[ind+1:]+'view'
        if name == 'viewview':
            name = view[:ind][view[:ind].rfind('.')+1:]+'view'
        val = _without_cls(view, name)
        if val is not None:
            return val
    except Exception: # pylint: disable=broad-except
        pass

    try:
        viewmod  = __import__(view[:view.rfind('.')],
                              fromlist = view[view.rfind('.')+1:])
    except ImportError:
        viewmod  = __import__('view.'+view[:view.rfind('.')],
                              fromlist = view[view.rfind('.')+1:])
    return getattr(viewmod, view[view.rfind('.')+1:])

def _debug(raiseerr, nothreading):
    # pylint: disable=import-outside-toplevel
    if nothreading:
        import view.base as _base
        _base.SINGLE_THREAD = True

    if raiseerr:
        from app.maincontrol import DisplayController
        DisplayController.CATCHERROR = False

def _launch(filtr, view, app, gui, kwa):
    viewcls = _from_path(view)
    if viewcls is None:
        viewcls = _from_module(view)

    app            = filtr(app, viewcls)
    kwa['runtime'] = gui
    if '.' in app and 'A' <= app[app.rfind('.')+1] <= 'Z':
        mod  = app[:app.rfind('.')]
        attr = app[app.rfind('.')+1:]
        launchmod = getattr(__import__(mod, fromlist = [attr]), attr)
    else:
        launchmod = __import__(app, fromlist = ['launch'])

    return getattr(launchmod, 'launch')(viewcls, **kwa)

CONFIGS = 'display', 'theme'
def _config(lines):
    if len(lines) == 0:
        return

    def _update_with_ctrl(ctrl, args, val):
        names, attrs = args.split("@")
        if not any(names.startswith(i) for i in CONFIGS):
            names = "theme."+names

        cnf   = getattr(ctrl, names[:names.find('.')])
        names = names[names.find('.')+1:]
        if '.' in attrs:
            attrs = attrs.split('.')
            mdl   = cur = copy(cnf.get(names, attrs[0]))
            for i in attrs[1:-1]:
                cur = getattr(cur, i)
            setattr(cur, attrs[-1], type(getattr(cur, attrs[-1]))(val))
            cnf.update(names, **{attrs[0]: mdl})
        else:
            mdl = type(cnf.get(names, attrs))(val)
            cnf.update(names, **{attrs: mdl})

    def _update_directly(ctrl, args, val):
        attrs = args.split('.')
        cur   = ctrl
        for i in attrs[:-1]:
            cur = getattr(cur, i)
        setattr(cur, attrs[-1], val)

    def _fcn(ctrl):
        for line in lines:
            if line == "clear":
                for key, vals in ctrl.theme.config.items():
                    chg = {i: deepcopy(vals.maps[1][i]) for i in vals.maps[0]}
                    if len(chg):
                        ctrl.theme.update(key, **chg)
                continue

            if '@' not in line and '=' not in line:
                # shortcut for selecting a tab
                line = "theme.app.tabs@initial="+line

            args,  val = line.split('=')
            if '@' in args:
                _update_with_ctrl(ctrl, args, val)
            else:
                _update_directly(ctrl, args, val)

    INITIAL_ORDERS.default_config = _fcn

def _version(softname, ctx, _, value):
    import version  # pylint: disable=import-outside-toplevel
    if not value or ctx.resilient_parsing:
        return
    click.echo(f'{softname} ' + version.version())
    click.echo(' - git hash:   ' + version.lasthash())
    click.echo(' - created on: ' + version.hashdate())
    click.echo(' - compiler:   ' + version.compiler())
    ctx.exit()

def defaultinit(config, wall, raiseerr, nothreading):
    "default start"
    if wall:
        WARNINGS.set(True)

    @INITIAL_ORDERS.append
    def _started(_, start = time()):
        LOGS.info("done loading in %d seconds", time()-start)

    _config(config)
    _debug(raiseerr, nothreading)

# pylint: disable=too-many-arguments
def defaultmain(filtr, view, gui, port, defaultapp):
    "Launches an view"
    kwargs = dict(port = port, apponly = False)
    server = _launch(filtr, view, defaultapp, gui, kwargs)
    server.run_until_shutdown()
    logging.shutdown()

def defaultclick(softname, *others, defaultview = None):
    """
    sets default command line options
    """
    def _wrapper(fcn):
        fcn = click.option("--nothreading",
                           flag_value = True,
                           default    = False,
                           help       = '[DEBUG] Runs plots in single thread')(fcn)
        fcn = click.option("--raiseerr",
                           flag_value = True,
                           default    = False,
                           help       = '[DEBUG] Whether errors should be caught')(fcn)
        fcn = click.option('-p', "--port",
                           default    = str(DEFAULT_SERVER_PORT),
                           help       = 'Port used: use "random" for any')(fcn)
        fcn = click.option("-g", "--gui",
                           type       = click.Choice(['app', 'browser', 'default',
                                                      'firefox-browser', 'chrome-browser',
                                                      'firefox-app', 'chrome-app',
                                                      'none']),
                           default    = 'app',
                           help       = 'The type of browser to use.')(fcn)
        fcn = click.option("-c", "--config",
                           multiple   = True,
                           help       = 'Changing the config')(fcn)
        fcn = click.option("-w", "--wall",
                           flag_value = True,
                           default    =  False,
                           help       = 'sets warnings as errors')(fcn)

        for i in others:
            fcn = i(fcn)

        fcn = click.argument('view', default = defaultview)(fcn)
        fcn = click.option('--version',
                           is_flag      = True,
                           callback     = partial(_version, softname),
                           expose_value = False,
                           is_eager     = True
                          )(fcn)

        return click.command()(fcn)
    return _wrapper

class Warnings:
    "sets warnings"
    def __init__(self):
        self._old = None

    def set(self, yes: bool):
        "sets the warnings"
        if yes:
            warnings.filterwarnings('error', category = RuntimeWarning, message = ".*All-NaN.*")
            warnings.filterwarnings('error', category = FutureWarning)
            warnings.filterwarnings('error', category = DeprecationWarning)
            warnings.filterwarnings('error', category = PendingDeprecationWarning)
            warnings.filterwarnings('ignore', category = DeprecationWarning,
                                    message  = '.*generator .* raised StopIteration.*')
            self._old = np.seterr(all='raise')
            return
        old, self._old = self._old, None
        if old:
            np.seterr(**old)
            warnings.resetwarnings()

WARNINGS = Warnings()
