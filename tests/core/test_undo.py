#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"All undo-related stuff"
from typing                 import Callable
from undo.view              import UndoView
from undo.control           import UndoController
from undo.model             import UndoModel
from control.decentralized  import DecentralizedController

class _KeyStrokes(dict):
    name = 'keystroke'

class _Model:
    def __init__(self):
        self.name    = "dummy"
        self.left    = []
        self.right   = []

class _BaseControl:
    wrapper: Callable
    def __init__(self):
        self.mdl     = UndoModel()
        self.undos   = UndoController(undos = self.mdl)
        self.theme   = DecentralizedController()
        self.display = DecentralizedController()
        self.display.add(_KeyStrokes())
        self.theme.add(_KeyStrokes())
        self.theme.add(_Model())

    def __undos__(self, wrapper):
        self.theme.__undos__(wrapper)
        self.display.__undos__(wrapper)

def test_undo():
    "test undo"
    ctrl =_BaseControl()
    mdl  = ctrl.mdl
    UndoView(ctrl).observe(ctrl)

    ctrl.display.handle("startaction", args = {'recursive': False})
    ctrl.display.handle("stopaction", args = {'recursive': False})
    assert not mdl.undos
    assert not mdl.redos
    ctrl.display.handle("startaction", args = {'recursive': False})
    ctrl.theme.update("dummy", left = [1])
    ctrl.display.handle("stopaction", args = {'recursive': False})
    assert len(mdl.undos) == 1
    assert not mdl.redos
    assert ctrl.theme.model("dummy").left == [1]

    ctrl.display.handle("startaction", args = {'recursive': False})
    ctrl.undos.undo()
    ctrl.display.handle("stopaction", args = {'recursive': False})
    assert len(mdl.redos) == 1
    assert not mdl.undos
    assert ctrl.theme.model("dummy").left == []

    ctrl.display.handle("startaction", args = {'recursive': False})
    ctrl.undos.redo()
    ctrl.display.handle("stopaction", args = {'recursive': False})
    assert len(mdl.undos) == 1
    assert not mdl.redos
    assert ctrl.theme.model("dummy").left == [1]

if __name__ == '__main__':
    test_undo()
