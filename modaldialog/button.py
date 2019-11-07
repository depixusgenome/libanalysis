#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Display the status of running jobs"
from abc                import abstractmethod
from copy               import deepcopy
from contextlib         import contextmanager
from dataclasses        import dataclass
from functools          import partial
from typing             import List, TypeVar, Generic
from bokeh.models       import Button
from bokeh.document     import Document

from utils.logconfig    import getLogger
from utils.inspection   import templateattribute
from .                  import dialog
from .builder           import tohtml

@dataclass
class DialogButtonConfig:
    "storage config"
    name:   str
    label:  str
    icon:   str = 'cog'
    width:  int = 100
    height: int = 28


LOGS  = getLogger(__name__)
Model = TypeVar("Model")
Theme = TypeVar("Theme", bound = DialogButtonConfig)


class ModalDialogButton(Generic[Theme, Model]):
    "explore current storage"
    _widget:  Button
    _doc:     Document

    def __init__(self):
        self._theme = templateattribute(self, 0)()

    def swapmodels(self, ctrl):
        "swap with models in the controller"
        self._theme = ctrl.theme.swapmodels(self._theme)

        for i in self.__dict__.values():
            if hasattr(i, 'swapmodels'):
                i.swapmodels(ctrl)

    def addtodoc(self, ctrl, doc) -> List[Button]:
        "creates the widget"
        icon = None
        if self._theme.icon:
            try:
                # import here to reduce dependencies
                from view.fonticon import FontIcon
                icon  = FontIcon(iconname = self._theme.icon)
                label = self._theme.label
            except ImportError:
                label = self._theme.label if self._theme.label else self._theme.icon
                icon  = None

        self._widget = Button(
            width  = self._theme.width,
            height = self._theme.height,
            label  = label,
            icon   = icon
        )

        self._widget.on_click(partial(self.run, ctrl, doc))

        return [self._widget]

    def run(self, ctrl, doc):
        "method to trigger the modal dialog"
        try:
            current = self._newmodel(ctrl)
            default = self._defaultmodel(ctrl, current)
            return dialog(
                doc,
                **tohtml(self._body(current), current, default),
                context = partial(self._onsubmit, ctrl, deepcopy(current), current),
                model   = current,
                always  = True
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGS.exception(exc)
            tohtml(f"ERROR: {exc}")

    @contextmanager
    def _onsubmit(self, ctrl, current, changed, _):
        yield

        diff = self._diff(current, changed)
        if not diff:
            return

        with ctrl.action:
            self._action(ctrl, diff)

    def _diff(self, current: Model, changed: Model):
        if hasattr(current, 'diff'):
            return getattr(current, 'diff')(changed)
        raise NotImplementedError(f"{self}._diff")

    def _newmodel(self, ctrl) -> Model:
        return templateattribute(self, 1)(self, ctrl)

    @staticmethod
    def _defaultmodel(_, current) -> Model:
        return current

    @abstractmethod
    def _body(self, current: Model) -> str:
        return ""

    @abstractmethod
    def _action(self, ctrl, diff):
        return ""
