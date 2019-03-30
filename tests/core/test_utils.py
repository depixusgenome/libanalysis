#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"Test utils"
# pylint: disable=import-error
# pylint: disable=no-self-use
from   enum               import Enum
from   typing             import Generic, TypeVar, Tuple
from   functools          import partial
import pathlib
import pytest
import numpy as np
from   utils              import escapenans, fromstream
from   utils.gui          import intlistsummary, parseints
from   utils.lazy         import LazyInstError, LazyInstanciator, LazyDict
from   utils.attrdefaults import fieldnames, changefields, initdefaults
from   utils.inspection   import templateattribute, diffobj, isfunction, ismethod, parametercount

class TestLazy:
    u"test lazy stuff"
    def test_instanciator_verify(self):
        u"test instanciator argument verifications"
        with pytest.raises(LazyInstError):
            LazyInstError.verify(lambda x: None)
            LazyInstError.verify(1)
            LazyInstError.verify("")
        assert LazyInstError.verify(lambda *x, y = 1, **z: None) is None
        assert LazyInstError.verify(lambda y = 1, **z: None)     is None
        assert LazyInstError.verify(lambda **z: None)            is None

    def test_instanciator(self):
        u"test instanciator"
        lst = []
        class _AnyType:
            def __init__(self):
                lst.append(1)

        fcn  = lambda: _AnyType() # pylint: disable=unnecessary-lambda
        lazy = LazyInstanciator(fcn)
        assert len(lst) == 0

        ans  = lazy()
        assert len(lst) == 1

        ans2 = lazy()
        assert len(lst) == 1
        assert ans is ans2

    def test_lazydict(self):
        u"test lazydict"
        lst = []

        def _create(name):
            def __init__(_):
                lst.append(name)
            return name, type(name, tuple(), dict(__init__ = __init__))

        for fcn in (iter, dict):
            dico = LazyDict(fcn((_create('l1'), _create('l2'))),
                            **dict((_create('l3'), _create('l4'))))
            assert len(lst) == 0

            assert dico['l1'].__class__.__name__    == 'l1'
            assert lst                              == ['l1']

            assert dico['l1'].__class__.__name__    == 'l1'
            assert lst                              == ['l1']

            assert 'l2' in dico
            del dico['l2']
            assert 'l2' not in dico

            assert lst == ['l1']

            assert dico.pop('l3').__class__.__name__    == 'l3'
            assert lst                                  == ['l1', 'l3']

            assert dico.pop('l3', lambda:111)           ==  111
            assert lst                                  == ['l1', 'l3']

            assert dico.get(*_create('l7')).__class__.__name__ == 'l7'
            assert lst                                         == ['l1', 'l3', 'l7']
            assert 'l7' not in dico

            assert dico.setdefault('l4', None).__class__.__name__ == 'l4'
            assert lst                                            == ['l1', 'l3', 'l7', 'l4']
            assert dico.setdefault('l4', None).__class__.__name__ == 'l4'
            assert lst                                            == ['l1', 'l3', 'l7', 'l4']

            assert dico.setdefault(*_create('l8')).__class__.__name__ == 'l8'
            assert lst == ['l1', 'l3', 'l7', 'l4', 'l8']
            assert 'l8' in dico
            assert dico.setdefault('l8', None).__class__.__name__ == 'l8'
            assert lst == ['l1', 'l3', 'l7', 'l4', 'l8']
            assert 'l8' in dico
            lst.clear()

def test_templateattrs():
    "test template attrs"
    TVar1 = TypeVar("TVar1")
    TVar2 = TypeVar("TVar2")
    class _AClass(Generic[TVar1]):
        pass

    class _BClass(_AClass[int]):
        pass
    assert templateattribute(_BClass, 0) is int

    class _AClass(Generic[TVar1, TVar2]):
        pass
    class _BClass(_AClass[int, Tuple[float,...]]): # type: ignore
        pass
    assert templateattribute(_BClass, 0) is int
    assert templateattribute(_BClass, 1) == Tuple[float, ...]

    class _BClass(_AClass[int, TVar2]): # type: ignore
        pass

    class _CClass(_BClass[float]): # type: ignore
        pass

    assert templateattribute(_CClass, 0) is float
    assert templateattribute(_CClass.__base__, 0) is int

def test_diffobj():
    "test template attrs"
    # pylint: disable=invalid-name,attribute-defined-outside-init
    raises = pytest.raises
    with raises(TypeError):
        diffobj((), [])

    left  = {'a': 1, 'b': 2, 'c': 3}
    right = {'a': 1, 'b': 2, 'c': 3}
    assert diffobj(left, right) == {}

    right['d'] = 1
    assert diffobj(left, right) == {}

    right.pop('c')
    with raises(KeyError):
        diffobj(left, right)

    right['c'] = 2
    assert diffobj(left, right) == {"c": 3}

    class _AClass:
        def __init__(self, info):
            self.__dict__.update(info)
    right = _AClass(left)
    left  = _AClass(left)
    assert diffobj(left, right) == {}

    right.d = 1
    assert diffobj(left, right) == {}

    del right.c
    with raises(KeyError):
        diffobj(left, right)

    right.c = 2
    assert diffobj(left, right) == {"c": 3}

    class _AClass:
        def __init__(self, info):
            self.__dict__.update(info)
        def __getstate__(self):
            xx =  dict(self.__dict__)
            xx.pop('c', None)
            return xx

    right = _AClass(left.__dict__)
    left  = _AClass(left.__dict__)
    assert diffobj(left, right) == {}
    right.d = 1
    assert diffobj(left, right) == {}
    del right.c
    assert diffobj(left, right) == {}
    right.c = 2
    assert diffobj(left, right) == {}
    right.b = 1
    assert diffobj(left, right) == {'b': 2}

    class _AClass:
        def __getstate__(self):
            return ()
    with raises(NotImplementedError):
        diffobj(_AClass(), _AClass())

def test_inspection():
    "test inspection"
    def _afcn():
        pass
    class _AClass:
        # pylint: disable=missing-docstring
        def aaa(self):
            pass
        @classmethod
        def bbb(cls):
            pass
        @staticmethod
        def ccc():
            pass
        def __call__(self, _):
            pass

    fcns = [
        _afcn, lambda x: None, _AClass.aaa, _AClass().aaa,
        _AClass.bbb, _AClass().bbb, _AClass.ccc, _AClass().ccc,
        partial(lambda x: None, 1)
    ]
    assert all(isfunction(i) for i in fcns)
    assert not isfunction(_AClass)
    assert not isfunction(_AClass())

    assert parametercount(lambda x: None) == 1
    assert parametercount(lambda x = 1: None) == 0
    assert parametercount(lambda: None) == 0
    assert parametercount(lambda x, **y: None) == 1
    assert parametercount(lambda x = 1, **y: None) == 0
    assert parametercount(lambda **y: None) == 0
    assert parametercount(lambda x, *y: None) == 1
    assert parametercount(lambda x = 1, *y: None) == 0
    assert parametercount(lambda *y: None) == 0

    assert ismethod(_AClass.aaa)
    assert not ismethod(_AClass().aaa)
    assert not ismethod(_AClass().bbb)
    assert not ismethod(_AClass.ccc)
    assert not ismethod(_AClass().ccc)
    assert not ismethod(_AClass.bbb)

def test_attrs():
    "test default attributes"
    class _Enum(Enum):
        aaa = 'aaa'
        bbb = 'bbb'

    class _AClass:
        aval: int   = 1
        bval: list  = [1]
        cval: _Enum = _Enum("aaa")
        @initdefaults(frozenset(locals()))
        def __init__(self, **_):
            pass
    assert _AClass().aval == 1
    assert _AClass().bval == [1]
    assert _AClass().bval is not _AClass.bval
    assert _AClass().cval == _Enum('aaa')
    assert _AClass(aval = 2).aval == 2
    assert _AClass(cval = 'bbb').cval == _Enum('bbb')
    bval= [1,2, 3]
    assert _AClass(bval = bval).bval is bval

    class _BClass(_AClass):
        aval: int = 3
    assert _BClass().aval == 3

    class _CClass(_AClass):
        aval: int = 4
        dval: tuple = (3, 3)
        fval: np.ndarray = np.array([3, 3])
        @initdefaults('dval', 'fval')
        def __init__(self, **_):
            super().__init__(**_)
    assert _CClass().aval == 4
    assert _CClass().dval == (3, 3)
    assert _CClass().dval is _CClass.dval
    assert _CClass().fval is not _CClass.fval
    assert isinstance(_CClass().fval, np.ndarray)
    assert list(_CClass().fval) == [3, 3]

def test_escapenans():
    u"tests that we can remove and add nans again"
    array1, array2 = np.arange(10)*1., np.ones((10,))*1.
    array1[[0,5]] = np.nan
    array2[[1,6]] = np.nan
    with escapenans(array1, array2) as (cur1, cur2):
        assert not any(np.isnan(cur1)) and not any(np.isnan(cur2))
        cur1 += cur2
    inds = [2,3,4,7,8,9]
    assert all(array1[inds] ==  (np.arange(10)+ np.ones((10,)))[inds])
    assert all(np.isnan(array1[[0,5]]))
    assert all(array1[[1,6]] == [1,6])
    assert all(np.isnan(array2[[1,6]]))
    assert all(array2[[0,5]] == 1)

def test_fromstream():
    u"tests fromstream"
    with open("/tmp/__utils__test.txt", "w") as stream:
        print("found", file = stream)

    @fromstream("r")
    def _read(dummy, self, path, another): # pylint: disable=unused-argument
        line = path.readlines()
        assert len(line) == 1
        assert line[0].startswith('found')

    _read(1, 2, pathlib.Path('/tmp/__utils__test.txt'), 3)
    _read(1, 2, '/tmp/__utils__test.txt', 3)
    with open('/tmp/__utils__test.txt', 'r') as stream:
        _read(1, 2, stream, 3)

def test_fieldnames():
    u"tests field names"
    # pylint: disable=missing-docstring,invalid-name
    class DescriptorSet:
        def __set__(self, *_):
            return 1

    class DescriptorGet:
        def __get__(self, *_):
            return 2

    class A:
        def __init__(self):
            self._name = 1
            self.name  = 2

        descget = DescriptorGet()
        descset = DescriptorSet()
        propset = property(None, lambda *_: None)
        propget = property(lambda _: 1)

        _descget = DescriptorGet()
        _descset = DescriptorSet()
        _propset = property(None, lambda *_: None)
        _propget = property(lambda _: 1)
    assert fieldnames(A()) == {'name', 'descset', 'propset'}

def test_changefields():
    "tests changefields"
    class _Aaa:
        def __init__(self):
            self.attr = 1

    try:
        val = _Aaa()
        with changefields(val, attr = 0):
            assert val.attr == 0
            raise KeyError()
    except KeyError:
        assert val.attr == 1

def test_init():
    "tests initdefaults"
    # pylint: disable=blacklisted-name
    class _Aaa:
        toto = 1.
        titi = [2.]
        @initdefaults(frozenset(locals()))
        def __init__(self, **_):
            pass

    class _Bbb:
        aaa  = tuple()
        tata = tuple()
        tutu = _Aaa()
        @initdefaults(frozenset(locals()),
                      tata = 'ignore',
                      tutu = 'update',
                      mmm  = lambda self, val: setattr(self, 'tata', val))
        def __init__(self, **_):
            pass

    assert _Aaa().titi is not _Aaa.titi
    assert _Aaa().titi == [2.]
    assert _Aaa().toto == 1.
    xxx = [3.]
    assert _Aaa(titi = xxx).titi is xxx

    assert _Bbb().tutu is not _Bbb.tutu
    assert _Bbb(tata = (1,)).tata == tuple()
    yyy = _Aaa()
    assert _Bbb(tutu = yyy).tutu is yyy
    assert _Bbb(toto = 2).tutu.toto == 2.

    _Bbb(tutu = yyy, toto = 2)
    assert yyy.toto == 2.

    assert _Bbb(mmm = xxx).tata is xxx

    # pylint: disable=missing-docstring
    class Cls:
        attr       = []
        ignored    = 0
        _protected = 1
        @initdefaults(frozenset(locals()),
                      ignored   = 'ignore',
                      protected = '_protected',
                      call      = lambda self, value: setattr(self, 'ignored', 2*value))
        def __init__(self, **kwa):
            pass

    assert Cls().ignored == 0
    assert Cls(call = 1).ignored == 2
    assert Cls().attr    == []
    assert Cls().attr    is not Cls.attr
    assert Cls()._protected == 1                # pylint: disable=protected-access
    assert Cls(protected = 2)._protected == 2   # pylint: disable=protected-access
    lst = [2]
    assert Cls(attr = lst).attr is lst

    class Trans:
        attr1 = 1
        attr2 = 2
        @initdefaults(frozenset(locals()))
        def __init__(self, *kwa:dict, **_) -> None:
            kwa[0].pop('attr1', None)
            if 'attr2' in kwa[0]:
                kwa[0]['attr2'] *= 2

    assert Trans(attr1 = 100).attr1 == 1
    assert Trans(attr2 = 100).attr2 == 200

    class Agg:
        elem = Cls()
        @initdefaults(frozenset(locals()), elem = 'update')
        def __init__(self, **kwa):
            pass

    assert Agg(attr = [2]).elem.attr == [2]

def test_intlist():
    "tests intlist"
    # pylint: disable=blacklisted-name
    assert intlistsummary(range(5)) == "0 → 4"
    assert intlistsummary(range(5,0, -1)) == "1 → 5"
    assert intlistsummary([])  == ""
    assert intlistsummary([1,2,3, 6,7, 10,11,12, 14]) == "1 → 3, 6, 7, 10 → 12, 14"
    assert intlistsummary([1,3, 6, 2, 10,11,7,12, 14]) == "1 → 3, 6, 7, 10 → 12, 14"

    assert parseints("0 → 4") == {0, 1, 2, 3,  4}
    assert parseints("1 → 5") == {1, 2, 3, 4, 5}
    assert parseints("") == set([])
    assert parseints("1 → 3, 6, 7, 10 → 12, 14")   == {1,2, 3, 6, 7, 10, 11, 12, 14}
    assert parseints("10 → 12, 1 → 3, 7 , 6 , 14") == {1,2, 3, 6, 7, 10, 11, 12, 14}

if __name__ == '__main__':
    test_inspection()