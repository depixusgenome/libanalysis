#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"all static view aspects here"
import sys
from   pathlib     import Path
from   tornado     import gen
from   tornado.web import StaticFileHandler, RequestHandler

DOCROUTE = "doc"
ROUTE    = 'view'
JQUERY   = [ROUTE+"/jquery.min.js", ROUTE+"/jquery-ui.min.js"]
if 'app.scripting' in sys.modules:
    sys.modules['app.scripting'].addload(__name__)  # type: ignore

def server(kwa):
    "adds a router to the server"
    router = ("/%s/(.*)" % ROUTE, StaticFileHandler, {"path": "static"})
    kwa.setdefault('extra_patterns', []).append(router)

    path   = next((i for i in ("doc", "../doc") if Path(i).exists()), "doc")
    router = ("/%s/(.*)" % DOCROUTE, StaticFileHandler, {"path": path})
    kwa.setdefault('extra_patterns', []).append(router)

def route(*names:str):
    "return the route to a given file"
    if len(names) == 0:
        return JQUERY

    out = []
    for name in names:
        path = Path("./static")/name
        out.append(
            ROUTE
            + "/" + name
            + (f"?v={int(path.stat().st_mtime)}" if path.exists() else "")
        )
    return out

class StaticHandler(RequestHandler):    # pylint: disable=abstract-method
    'Implements a custom Tornado handler for autoloading css'
    def initialize(self, *_, **__):  # pylint: disable=arguments-differ
        "init"

    @gen.coroutine
    def get(self, *_, **__):
        "getter"
        self.set_header("Content-Type", 'text/css')
        self.write(".dpx-peaksplot-widget { width: 520px;}\n"
                   ".dpx-peaksplot-canvas { width: auto;}\n")
