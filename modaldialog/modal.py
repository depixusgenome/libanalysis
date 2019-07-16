#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Allows creating modals from anywhere
"""
from    typing                  import Optional, Callable, Union, Sequence, Any

import  bokeh.core.properties   as props
from    bokeh.models            import Model, Callback
from    view.static             import ROUTE
from    .options                import tohtml, fromhtml

class DpxModal(Model):
    "Modal dialog"
    __css__            = [ROUTE+"/backbone.modal.css?v=2",
                          ROUTE+"/backbone.modal.theme.css"]
    __javascript__     = [ROUTE+"/underscore-min.js",
                          ROUTE+"/jquery.min.js"]
    __implementation__ = "modal.ts"
    title              = props.String("")
    body               = props.String("")
    buttons            = props.String("")
    results            = props.Dict(props.String, props.Any)
    submitted          = props.Int(0)
    startdisplay       = props.Int(0)
    keycontrol         = props.Bool(True)
    callback           = props.Instance(Callback)
    def __init__(self, **kwa):
        super().__init__(**kwa)
        self.__handler: Optional[Callable] = None
        self.__running = False
        self.__always  = False

        self.on_change('submitted', self._onsubmitted_cb)
        self.on_change('results',   self._onresults_cb)

    def run(self,                                       # pylint: disable=too-many-arguments
            title   : str                       = "",
            body    : Union[Sequence[str],str]  = "",
            callback: Callback                  = None,
            context : Callable[[str], Any]      = None,
            model                               = None,
            buttons                             = "",
            always                              = False):
        "runs the modal dialog"
        self.__handler = (
            None if isinstance(callback, Callback) or model is None else
            (lambda x: fromhtml(x, body, model)) if context is None else
            (lambda x: fromhtml(x, body, model, lambda : context(title))) # type: ignore
        )
        self.__always  = always
        self.__running = False
        self.update(title    = title,
                    body     = tohtml(body, model),
                    callback = callback,
                    buttons  = buttons,
                    results  = {})
        self.__running      = True
        self.startdisplay   = self.startdisplay+1

    def _onresults_cb(self, attr, old, new):
        pass

    def _onsubmitted_cb(self, attr, old, new):
        if self.__running and self.__handler and (self.__always or len(self.results)):
            self.__handler(self.results)

        self.__running = False
