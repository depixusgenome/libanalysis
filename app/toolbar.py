#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Updates app manager so as to deal with controllers and toolbar"
from typing           import Generic, TypeVar, cast

from bokeh.layouts    import layout, column
from utils.inspection import getclass, templateattribute

TOOLBAR = TypeVar("TOOLBAR")
VIEW    = TypeVar("VIEW")

class _AppName:
    def __get__(self, _, owner):
        cls  = templateattribute(owner, 1)
        assert cls is not None and not isinstance(cls, cast(type, TypeVar))
        return getattr(cls, "APPNAME", cls.__name__.replace('view', ''))

class ViewWithToolbar(Generic[TOOLBAR, VIEW]):
    "A view with the toolbar on top"
    APPNAME = _AppName()
    _bar:      TOOLBAR
    _mainview: VIEW

    def __init__(self, ctrl = None, **kwa):
        assert not isinstance(templateattribute(self, 0), cast(type, TypeVar))
        self._bar = templateattribute(self, 0)(ctrl = ctrl, **kwa)

        theme     = ctrl.theme.model("theme")
        height    = min(
            theme.tabheight,
            theme.appsize[1] - self._bar.gettoolbarheight() - theme.borders
        )
        ctrl.theme.updatedefaults("theme", tabheight = height)
        ctrl.theme.update("theme", tabheight = height)
        self._mainview  = templateattribute(self, 1)(ctrl = ctrl, **kwa)

    def swapmodels(self, ctrl):
        "swap models with those in the controller"
        for i in (self._bar, self._mainview):
            if callable(getattr(i, 'swapmodels', None)):
                i.swapmodels(ctrl)

    def ismain(self, ctrl):
        "sets-up the main view as main"
        getattr(self._mainview, 'ismain', lambda _: None)(ctrl)

    def close(self):
        "remove controller"
        self._bar.close()
        self._mainview.close()

    def observe(self, ctrl):
        "observe the controller"
        self._bar.observe(ctrl)
        self._mainview.observe(ctrl)

    def addtodoc(self, ctrl, doc):
        "adds items to doc"
        tbar    = self._bar.addtodoc(ctrl, doc)
        others  = self._mainview.addtodoc(ctrl, doc)
        appsize = ctrl.theme.get("theme", "appsize")
        while isinstance(others, (tuple, list)) and len(others) == 1:
            others = others[0]

        if isinstance(others, list):
            children = [tbar] + others
        elif isinstance(others, tuple):
            children = [tbar, layout(others, sizing_mode = 'stretch_both')]
        else:
            children = [tbar, others]

        for i in children[1:]:
            i.sizing_mode = 'stretch_both'
        return column(
            children,
            sizing_mode = 'stretch_both',
            css_classes = ["dpx-tb-layout"],
            width       = appsize[0],
            height      = appsize[1],
        )

def toolbarview(tbar, main) -> type:
    "return the view with toolbar"
    cls = getclass(tbar), getclass(main)   # pylint: disable=unused-variable

    class ToolbarView(ViewWithToolbar[cls]):  # type: ignore
        "Toolbar view"
    return ToolbarView
