#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Testing anastore"
import sys
import numpy

import anastore
import anastore.api
from   anastore      import modifyclasses, TPE, DELETE, RESET

BEENTHERE = []
class _CycleCreatorTask:
    def __init__(self, **kwa):
        self.__dict__.update(kwa)

class _TrackReaderTask:
    def __init__(self, **kwa):
        self.__dict__.update(kwa)

class _TaggingTask:
    def __init__(self, **kwa):
        self.__dict__.update(kwa)

class _Toto(_TaggingTask):
    def __init__(self, **kwa):
        super().__init__(**kwa)
        self.attr = kwa['attr']

    def __getstate__(self):
        BEENTHERE.append(1)
        return tuple(self.__dict__.items())

    def __setstate__(self, args):
        BEENTHERE.append(2)
        self.__dict__.update(self.__class__(**dict(args)).__dict__)

def test_storetasks(monkeypatch):
    u"tests storing tasks"
    mods = sys.modules
    for i in (_TrackReaderTask, _CycleCreatorTask, _TaggingTask, _Toto):
        for mod in ('__main__', 'anastoretest' ):
            monkeypatch.setattr(mods.get(mod, mods['__main__']), i.__name__, i, False)

    tasks = [_TrackReaderTask(path = 'mypath'),
             _TaggingTask(level  = 'bead',
                          tags   = {'tag1': {1,2,3}, 'tag3': {4, 5, 6}},
                          action = 'remove'),
             _CycleCreatorTask(),
             _TaggingTask(level  = 'cycle',
                          tags   = {'tag4': {(1,1),(2,1),(3,1)}},
                          action = 'keep'),
             _Toto(level  = 'peak',
                   tags   = {'tag5': {(1,1),(2,1),(3,1)}},
                   action = 'keep',
                   attr   = 'whatever')]

    used = []
    def _vers(ind):
        def _fcn(val, _ = ind):
            used.append(_)
            return val
        return _fcn

    patch = anastore.Patches()
    for _ in range(5):
        patch.patch(_vers(_))

    monkeypatch.setattr(anastore.api, 'PATCHES', {'tasks': patch})
    dumped = anastore.dumps(tasks)
    assert dumped[:len('[{"version": 5},')] == '[{"version": 5},'

    for _ in range(5,8):
        patch.patch(_vers(_))

    loaded = anastore.loads(dumped)
    assert used   == list(range(5, 8))
    assert BEENTHERE == [1,2]
    for i, j in zip(loaded, tasks):
        assert i.__dict__ == j.__dict__, str(i)

def test_storenumpy():
    u"tests storing arrays"
    vals = dict(a = numpy.array([None]*5), b = numpy.zeros((200,), dtype = numpy.int8),
                c = numpy.arange(5, dtype = numpy.float32),
                d = numpy.nanmedian)
    loaded = anastore.loads(anastore.dumps(vals))
    assert set(vals.keys()) == set(loaded.keys()) # pylint: disable=no-member
    assert numpy.array_equal(vals['a'], loaded['a'])
    assert numpy.array_equal(vals['b'], loaded['b'])
    assert numpy.array_equal(vals['c'], loaded['c'])
    assert loaded['d'] is numpy.nanmedian

def test_storetype(monkeypatch):
    u"tests storing arrays"
    mods = sys.modules
    for i in (_TaggingTask,):
        for mod in ('__main__', 'anastoretest' ):
            monkeypatch.setattr(mods.get(mod, mods['__main__']), i.__name__, i, False)
    vals = dict(a = _TaggingTask, b = [_TaggingTask])
    loaded = anastore.loads(anastore.dumps(vals))
    assert set(vals.keys()) == set(loaded.keys()) # pylint: disable=no-member
    assert vals['a'] is _TaggingTask
    assert vals['b'][0] is _TaggingTask

def test_modifyclass():
    "tests class modifications"
    val = [{TPE: 'toto'}, {TPE: 'titi', 'attr': 1}]
    modifyclasses(val, 'toto', DELETE, 'titi', RESET)
    assert val == [{TPE: 'titi'}]

    val = {'a':{TPE: 'toto'}, 'b': {TPE: 'titi', 'attr': 1}}
    modifyclasses(val, 'toto', DELETE, 'titi', RESET)
    assert val == {'b': {TPE: 'titi'}}

    val = [{'a':{TPE: 'toto', 'delete': 1, 'change': 2, 'reset' : 3, 'reset2': 4}}]
    modifyclasses(val, 'titi', dict(), 'toto',
                  dict(delete   = DELETE,
                       reset    = RESET,
                       reset2   = lambda x: RESET,
                       delete2  = lambda x: DELETE,
                       change   = lambda x: x*2,
                       __name__ = 'tata'))
    assert val == [{'a': {TPE: 'tata', 'change': 4}}]

    val = [{'a':{TPE: 'toto', 'attr': 1}}]
    modifyclasses(val, 'toto', dict(attr = 'newname'),
                  'toto', dict(newname = lambda x: x*2))
    assert val == [{'a': {TPE: 'toto', 'newname': 2}}]
