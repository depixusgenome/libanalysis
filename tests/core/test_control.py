#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Test control"
# pylint: disable=import-error,missing-docstring
from    typing                  import Callable, cast
from    control.event           import Event, EmitPolicy
from    control.decentralized   import DecentralizedController

def test_evt():
    "test event stuff"
    # pylint: disable=no-self-use,missing-docstring
    events = Event()

    calls  = []
    class _Ctrl:
        @staticmethod
        @events.emit(returns = EmitPolicy.inputs)
        def event1(*_1, **_2):
            calls.append("e1")
            return 1

        @classmethod
        @events.emit
        def event2(cls, *_1, **_2) -> dict:
            calls.append("e2")
            return dict(name = 'e2')

        @events.emit
        def event3(self, *_1, **_2) -> tuple:
            calls.append("e3")
            return ('e3',)

        @staticmethod
        @events.emit
        def event4(*_1, **_2) -> None:
            calls.append("e4")

    @events.emit('event5', 'event6', returns = EmitPolicy.inputs)
    def event5(*_1, **_2):
        calls.append("e5")
        return ('e5',)

    hdls = [] # type: ignore

    # pylint: disable=unused-argument
    class _Obs:
        @staticmethod
        @events.observe
        def onevent1(*args, calllater = None, **kwargs):
            assert (args, kwargs) == hdls[-1]

        @events.observe
        @staticmethod
        def onevent2(calllater = None, **kwargs):
            assert kwargs == dict(name = 'e2')

        @events.observe('event3')
        @staticmethod
        def onevent3(arg, calllater = None):
            assert arg == 'e3'

    got = []
    def _got(*args, calllater = None, **kwargs):
        got.append((args, kwargs))
    events.observe('event4', 'event6', _got)

    def onevent5(*args, calllater = None, **kwargs):
        assert (args, kwargs) == hdls[-1]

    events.observe(onevent5)

    ctrl = _Ctrl()
    obs  = _Obs() # pylint: disable=unused-variable

    hdls.append(((1,2,3), dict(tt = 2)))
    ctrl.event1(1,2,3, tt = 2)
    ctrl.event2(1,2,3, tt = 2)
    ctrl.event3(1,2,3, tt = 2)

    assert len(got) == 0
    ctrl.event4(1,2,3, tt = 2)

    assert got == [(tuple(), dict())]

    event5(1,2,3, tt = 2)
    assert got == [(tuple(), dict()),hdls[-1]]

def test_evt_observewithdict():
    "test event stuff"
    # pylint: disable=no-self-use,missing-docstring,unnecessary-lambda,multiple-statements
    events = Event()

    def _add(ind):
        # pylint: disable=unused-argument
        def _fcn(self, *_1, **_2):
            return dict(name = 'e%d' % ind)
        _fcn.__name__ = _fcn.__name__.replace('_fcn', 'event%d') % ind
        _fcn.__qualname__ = _fcn.__qualname__.replace('_fcn', 'event%d') % ind

        return _fcn.__name__, events.emit(_fcn, returns = EmitPolicy.outasdict)

    ctrl = type('_Ctrl', tuple(), dict(_add(i) for i in range(8)))()

    got  = []
    def onEvent3(name = None, **_):
        got.append(name)
    def _onEvent4(name = None, **_):
        got.append(name)
    def _onEvent6(name = None, **_):
        got.append(name)
    def _onEvent7(name = None, **_):
        got.append(name)

    events.observe({'event1': lambda **_: onEvent3(**_),
                    'event2': _onEvent4})
    events.observe(onEvent3, _onEvent4)
    events.observe(event5 = _onEvent4)
    events.observe([_onEvent6, _onEvent7])

    for i in range(1, 8):
        getattr(ctrl, 'event%d' % i)()
    assert got == ['e%d'% i for i in range(1, 8)]

def test_decentralized():
    "test decentralized"
    # pylint: disable=too-many-statements,missing-docstring
    class Toto:
        name = 'toto'
        aval = 1
        bval = ""
        def __init__(self, **_):
            self.aval = _.get('aval', 2)
            self.bval = _.get("bval", "")

    class Tata(dict):
        """
        Model for key bindings
        """
        name = 'toto'

    def _test(obj): # pylint: disable=too-many-statements
        ctrl = DecentralizedController()
        ctrl.add(obj)
        cnt = [0, 0]
        get = cast(Callable, dict.__getitem__  if isinstance(obj, dict) else getattr)
        def _fcn1(**_):
            cnt[0] += 1
        def _fcn2(**_):
            cnt[1] += 1
        ctrl.observe("totodefaults", _fcn1)
        ctrl.observe("toto",         _fcn2)
        cmap  = ctrl.config['toto']
        assert len(cmap.maps[0]) == 0
        assert cmap.maps[1] == {'aval': 2, 'bval': ""}

        ctrl.updatedefaults("toto", aval = 3)
        assert cnt == [1, 1]
        assert get(ctrl.model('toto'), 'aval') == 3
        assert get(ctrl.model('toto'), 'bval') == ""
        cmap  = ctrl.config['toto']
        assert len(cmap.maps[0]) == 0
        assert cmap.maps[1] == {'aval': 3, 'bval': ""}

        ctrl.updatedefaults("toto", aval = 3)
        assert cnt == [1, 1]

        ctrl.update("toto", aval = 3)
        assert cnt == [1, 1]

        ctrl.update("toto", aval = 4)
        assert cnt == [1, 2]
        assert get(ctrl.model('toto'), 'aval') == 4
        assert get(ctrl.model('toto'), 'bval') == ""
        cmap  = ctrl.config['toto']
        assert cmap.maps[0] == {'aval': 4}
        assert cmap.maps[1] == {'aval': 3, 'bval': ""}

        ctrl.updatedefaults("toto", aval = 6)
        assert cnt == [2, 2]
        assert get(ctrl.model('toto'), 'aval') == 4
        assert get(ctrl.model('toto'), 'bval') == ""
        cmap  = ctrl.config['toto']
        assert cmap.maps[0] == {'aval': 4}
        assert cmap.maps[1] == {'aval': 6, 'bval': ""}

        try:
            ctrl.update("toto", newval = 5)
        except KeyError:
            pass
        else:
            assert False

        if isinstance(obj, dict):
            ctrl.updatedefaults('toto', newval = 5)
            assert cnt == [3, 3]
            assert get(ctrl.model('toto'), 'aval') == 4
            assert get(ctrl.model('toto'), 'bval') == ""
            assert get(ctrl.model('toto'), 'newval') == 5
            cmap  = ctrl.config['toto']
            assert cmap.maps[0] == {'aval': 4}
            assert cmap.maps[1] == {'aval': 6, 'bval': "", 'newval': 5}

            ctrl.update('toto', newval = 10)
            assert cnt == [3, 4]
            assert get(ctrl.model('toto'), 'newval') == 10

            ctrl.updatedefaults('toto', newval = ctrl.DELETE)
            assert cnt == [4, 5]
            assert get(ctrl.model('toto'), 'aval') == 4
            assert get(ctrl.model('toto'), 'bval') == ""
            cmap  = ctrl.config['toto']
            assert cmap.maps[0] == {'aval': 4}
            assert cmap.maps[1] == {'aval': 6, 'bval': ""}
        else:
            try:
                ctrl.updatedefaults('toto', aval = ctrl.DELETE)
            except ValueError:
                pass
            else:
                assert False

        cnt[0] = cnt[1] = 0
        ctrl.update('toto', aval = ctrl.DELETE)
        assert cnt == [0, 1]
        assert get(ctrl.model('toto'), 'aval') == 6
        assert get(ctrl.model('toto'), 'bval') == ""
        cmap  = ctrl.config['toto']
        assert len(cmap.maps[0]) == 0
        assert cmap.maps[1] == {'aval': 6, 'bval': ""}

        with ctrl.localcontext(toto = dict(aval = 1)):
            assert get(ctrl.model('toto'), 'aval') == 1
            assert get(ctrl.model('toto'), 'bval') == ''
            ctrl.update("toto", bval = "***")
            assert get(ctrl.model('toto'), 'bval') == '***'
        assert get(ctrl.model('toto'), 'aval') == 6
        assert get(ctrl.model('toto'), 'bval') == ''


    _test(Toto())
    _test(Tata(aval = 2, bval = ""))

if __name__ == '__main__':
    test_evt()
