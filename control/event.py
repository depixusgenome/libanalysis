#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"Base event handler"
import re
import asyncio

from enum               import Enum, unique
from itertools          import product
from functools          import wraps, partial
from pathlib            import Path
from typing             import (
    Dict, Union, Sequence, Callable, Tuple, Any, Set, ClassVar, Pattern, Type,
    Iterable, Optional, List, FrozenSet,cast)

import pickle
from utils              import ismethod, isfunction, toenum
from utils.logconfig    import getLogger

LOGS         = getLogger(__name__)

_CNT         = [0]
_COMPLETIONS = Dict[Callable, Set[Pattern]]
_HANDLERS    = Dict[str, Union[Set[Callable], _COMPLETIONS]]
_OBSERVERS   = Union['HashFunc', Callable, staticmethod]

class NoEmission(Exception):
    "can be raised to stop an emission"

@unique
class EmitPolicy(Enum):
    "what elements to add to a fired event"
    outasdict   = 0
    outastuple  = 1
    inputs      = 2
    nothing     = 3
    annotations = 4

    @classmethod
    def get(cls, policy:'EmitPolicy', fcn) -> 'EmitPolicy':
        "returns the correct policy"
        if policy not in (cls.annotations, None):
            return policy

        if policy is cls.annotations:
            if isinstance(fcn, cast(type, staticmethod)):
                fcn = getattr(fcn, '__func__')
            try:
                rta = fcn.__annotations__['return']
            except KeyError as exc:
                raise KeyError("Missing emission policy: "+str(fcn)) from exc
        elif policy is None:
            rta = fcn if fcn is None or isinstance(fcn, type) else type(fcn)

        return (cls.nothing     if rta is None                      else
                cls.outasdict   if issubclass(rta, Dict)            else
                cls.outastuple  if issubclass(rta, (tuple, list))   else
                policy)

    def run(self, allfcns: Set[Callable], calldepth: int, args):
        "runs provided observers"
        calllater: List[Callable[[],None]] = []
        kwa       = dict(
            calllater = calllater,
            calldepth = calldepth
        )
        if   self is self.outasdict:
            dargs = cast(Dict, args)
            for hdl in allfcns:
                LOGS.debug("observer %s", hdl)
                hdl(**dargs, **kwa)
        elif self is self.outastuple:
            for hdl in allfcns:
                LOGS.debug("observer %s", hdl)
                hdl(*args, **kwa)
        elif self is self.nothing:
            for hdl in allfcns:
                LOGS.debug("observer %s", hdl)
                hdl(**kwa)
        else:
            for hdl in allfcns:
                LOGS.debug("observer %s", hdl)
                hdl(*args[0], **args[1], **kwa)

        for i in calllater:
            LOGS.debug("callater %s", i)
            i()

class EventHandlerContext:
    "handle a list of events repeatedly"
    _fcns:      Callable
    _policy:    EmitPolicy

    def __init__(self, ctrl, lst, policy, args):
        self._ctrl   = ctrl
        self._lst    = lst
        self._args   = policy, args

    def __enter__(self):
        self._fcns   = self._ctrl.getobservers(self._lst)
        self._policy = EmitPolicy.get(cast(EmitPolicy, self._args[0]), self._args[1])
        return self

    def __exit__(self, *_):
        pass

    def handle(self, args):
        "handle events"
        _CNT[0] += 1
        LOGS.debug("[%d] Handling %s (%s)", _CNT[0], self._lst, self._ctrl)
        self._policy.run(self._fcns, 0, args)
        LOGS.debug("[%d] Handled %s (%s)", _CNT[0], self._lst, self._ctrl)

    __call__ = handle

class HashFunc:
    """wrap a fonction such that its hash is the functon name & code line number"""
    __slots__ = ('_func', '_hash', '_str')
    __OBS_NAME: ClassVar[Pattern] = re.compile(r'^_*?on_*?(\w+)', re.IGNORECASE)
    __EM_NAME:  ClassVar[Pattern] = re.compile(r'^_*?(\w+)',     re.IGNORECASE)

    def __init__(self, fcn: _OBSERVERS, hashing = None):
        if isinstance(fcn, cast(type, staticmethod)):
            fcn = getattr(fcn, '__func__')

        elif ismethod(fcn):
            raise NotImplementedError(
                "observe cannot decorate a method unless it's a static one"
            )

        self._func = fcn

        if isinstance(fcn, HashFunc):
            self._func = getattr(fcn, '_func')
            self._hash = getattr(fcn, '_hash')
            self._str  = getattr(fcn, '_str')
            return

        if hasattr(hashing, '_hash'):
            self._hash = getattr(hashing, '_hash')
            self._str  = getattr(hashing, '_str')
            return

        hashf = self.callable(hashing if callable(hashing) else fcn)
        if not hasattr(hashf, '__qualname__'):
            hashf = getattr(hashf, '__call__')

        self._str  = (
            hashf.__qualname__,
            hashf.__code__.co_filename,
            hashf.__code__.co_firstlineno,
            hashing
        )
        self._hash = hash(self._str)

    def __repr__(self):
        return f"FWrap<{self._str[0]}@{Path(self._str[1]).stem}:{self._str[2]} [{self._hash}]"

    @classmethod
    def hashwith(cls, *args):
        "wraps a function in a HashFunc"
        def _wrapper(fcn):
            if isinstance(fcn, cls):
                fcn = getattr(fcn, '_func')
            return cls(fcn, hash(args) if args else None)
        return _wrapper

    @classmethod
    def funcname(cls, fcn: _OBSERVERS) -> str:
        "return the name of the method inside an object"
        return cls.callable(fcn).__name__

    @classmethod
    def eventname(cls, fcn: _OBSERVERS) -> str:
        "return the event name of the method inside an object"
        match = cls.__OBS_NAME.match(cls.funcname(fcn))
        return match.group(1).lower().strip() if match else ''

    @classmethod
    def emissionname(cls, fcn: _OBSERVERS) -> str:
        "return the event name of the method inside an object"
        match = cls.__EM_NAME.match(cls.funcname(fcn))
        if match is None:
            raise KeyError(f"Could not find emission name in {fcn}")
        return match.group(1).lower().strip()

    @classmethod
    def observable(cls, fcn) -> bool:
        "return whether the argument can be an observer"
        return isinstance(fcn, cls) or isfunction(fcn)

    @staticmethod
    def callable(fcn: _OBSERVERS) -> Callable:
        "return the callable inside an object"
        if isinstance(fcn, HashFunc):
            fcn = getattr(fcn, '_func')

        if isinstance(fcn, (classmethod, staticmethod)):
            fcn = fcn.__func__
        if isinstance(fcn, partial):
            fcn = fcn.func
        return cast(Callable, fcn)

    def __eq__(self, other):
        return other.__class__ is HashFunc and getattr(other, '_hash') == self._hash

    def __hash__(self):
        return self._hash

    def __call__(self, *args, **kwa):
        return self._func(*args, **kwa)

class Event:
    "Event handler class"
    emitpolicy: ClassVar[Type]    = EmitPolicy
    __SIMPLE:   ClassVar[Pattern] = re.compile(r'^(\w|\.)+$',   re.IGNORECASE)

    def __init__(self, **kwargs):
        self._handlers: _HANDLERS = kwargs.get('handlers', dict())
        self._calldepth           = [0]

    def linkdepths(self, ctrl):
        "links multiple controllers to the same call depths"
        self._calldepth = getattr(ctrl, '_calldepth')

    @staticmethod
    def hashwith(*args):
        "wraps a function in a HashFunc"
        return HashFunc.hashwith(*args)

    def remove(self, *args):
        "removes an event"
        if all(HashFunc.observable(i) for i in args):
            for arg in args:
                name = HashFunc.eventname(arg)
                assert name
                itm  = self._handlers.get(name, None)
                if isinstance(itm, Set):
                    tmp  = {arg,  HashFunc.callable(arg)}
                    itm -= {i for i in itm if i in tmp or getattr(i, 'func', None) in tmp}
            return

        name = args[0]
        assert isinstance(name, str)
        itm  = self._handlers.get(name, None)
        if isinstance(itm, Set):
            tmp  = set(args[1:]) | set(HashFunc.callable(i) for i in args[1:])
            itm -= {i for i in itm if i in tmp or getattr(i, 'func', None) in tmp}

    def getobservers(self, lst:Union[str,Set[str]]) -> Set[Callable]:
        "returns the list of observers"
        if isinstance(lst, str):
            lst = {lst}

        allfcns: Set[Callable] = set()
        for name in lst.intersection(self._handlers):
            allfcns.update(self._handlers[name])

        completions = self._handlers.get('ㄡ', None)
        if completions:
            for fcn, names in cast(_COMPLETIONS, completions).items():
                if any(patt.match(key) for patt, key in product(names, lst)):  # type: ignore
                    allfcns.add(fcn)
        return allfcns

    def handle(
            self,
            lst:    Union[str,Set[str]],
            policy: Optional[EmitPolicy]                 = None,
            args:   Optional[Union[Tuple,Sequence,Dict]] = None
    ):
        "Call handlers only once: collect them all"
        allfcns = self.getobservers(lst)
        if len(allfcns):
            _CNT[0] += 1
            policy = EmitPolicy.get(cast(EmitPolicy, policy), args)
            LOGS.debug("[%d] Handling %s (%s)", _CNT[0], lst, self)
            try:
                self._calldepth[0] += 1
                policy.run(allfcns, self._calldepth[0], args)
            finally:
                self._calldepth[0] -= 1
            LOGS.debug("[%d] Handled %s (%s)", _CNT[0], lst, self)
        return args

    def __call__(
            self,
            lst:    Union[str,Set[str]],
            policy: Optional[EmitPolicy]                 = None,
            args:   Optional[Union[Tuple,Sequence,Dict]] = None
    ):
        return EventHandlerContext(self, lst, policy, args)

    def emit(self, *names, returns = EmitPolicy.annotations):
        "wrapped methow will fire events named in arguments"
        def _wrapper(fcn:Callable, myrt = toenum(EmitPolicy, returns)):
            lst = self.__emit_list(names, fcn)

            return (self.__decorate_meth(self, lst, myrt, fcn) if ismethod(fcn) else
                    self.__decorate_func(self, lst, myrt, fcn))
        return self.__return(names, _wrapper)

    @classmethod
    def internalemit(cls, *names, returns = EmitPolicy.annotations):
        "wrapped methow will fire events named in arguments"
        def _wrapper(fcn:Callable, myrt = returns):
            lst = cls.__emit_list(names, fcn)
            return cls.__decorate_int(lst, myrt, fcn)

        return cls.__return(names, _wrapper)

    def observe(  # pylint: disable=too-many-arguments
            self,
            *anames,
            decorate: Optional[Callable[[Callable], Callable]] = None,
            argstest: Optional[Callable[..., bool]]            = None,
            **kwargs: Callable
    ):
        """
        Wrapped method will handle events named in arguments.

        This can be called directly:

        ```python
        event.observe('event 1', 'event 2',  observing_method)
        event.observe(onevent3)
        event.observe({'event1': fcn1, 'event2': fcn2})
        event.observe(event1 = fcn1, event2 = fcn2)
        ```

        or as a wrapper:

        ```python
        @event.observe('event 1', 'event 2')
        def observing_method(*args, **kwargs):
            pass

        @event.observe
        def onevent3(*args, **kwargs):
            pass
        ```
        """
        add = partial(
            self.__add_func,
            decorate = decorate,
            test     = argstest
        )

        def _fromfcn(fcn:Callable, name = None):
            name = name if name else HashFunc.eventname(fcn)
            assert name
            return add((name,), fcn)

        if len(anames) == 1:
            if hasattr(anames[0], 'items'):
                kwargs.update(cast(dict, anames[0]))
                names: Sequence[Any] = tuple()
            elif isinstance(anames[0], (list, tuple)):
                names = anames[0]
            else:
                names = anames
        else:
            names = anames

        if len(kwargs):
            for name, val in kwargs.items():
                _fromfcn(val, name)

        if len(names) == 0:
            return _fromfcn

        if all(isinstance(name, str) for name in names):
            def _wrapper(fcn):
                return add(names, fcn)
            return _wrapper

        if all(HashFunc.observable(name) for name in names):
            # dealing with tuples and lists
            for val in names[:-1]:
                _fromfcn(val)
            return _fromfcn(names[-1])

        return add(names[:-1], names[-1])

    class _OneShot:
        def __init__(self, hdls,  name, fcn):
            self._hdls = hdls
            self._name = name
            self._fcn  = fcn

        def __call__(self, *args, **kwa):
            fcn = self.discard()
            return fcn(*args, **kwa) if callable(fcn) else None

        def isdone(self) -> bool:
            "remove from handlers"
            return not len(self.__dict__)

        def discard(self) -> Optional[Callable]:
            "remove from handlers"
            if self.isdone():
                return None

            fcn = self._fcn
            self._hdls[self._name].discard(self)
            self.__dict__.clear()  # make sure the function cannot be called again
            return fcn

        def timeout(self, timeout: Optional[float]):
            "add a timeout on this observer"
            if timeout:
                async def _rem():
                    await asyncio.sleep(timeout)
                    self.discard()

                asyncio.create_task(_rem())
            return self

    def oneshot(self, name: str, fcn, timeout: Optional[float] = None):
        """
        one shot observation
        """
        name = name.lower().strip()
        shot = self._OneShot(self._handlers, name, fcn).timeout(timeout)
        self.__add_func([name], shot)
        assert shot in self._handlers[name]
        return shot

    def close(self):
        "Clear all handlers"
        self._handlers.clear()
        self._handlers = dict()

    @classmethod
    def __emit_list(cls, names, fcn = None) -> FrozenSet[str]:
        "creates a list of emissions"
        if len(names) == 0 or names[0] is fcn:
            names = (HashFunc.emissionname(fcn),)

        return frozenset(name.lower().strip() for name in names)

    @staticmethod
    def __handle_args(lst, policy, ret, args, kwargs):
        if policy in (EmitPolicy.outastuple, EmitPolicy.outasdict):
            return (lst, policy, ret)
        if policy == EmitPolicy.nothing:
            return (lst, policy)
        return (lst, policy, (args, kwargs))

    @staticmethod
    def __return(names, fcn):
        "Applies a wrapVper now or later"
        return fcn(names[0]) if len(names) == 1 else fcn

    @classmethod
    def __decorate_meth(cls, this, lst, myrt, fcn):
        "returns a decorator for wrapping methods"
        myrt = EmitPolicy.get(myrt, fcn)

        @wraps(fcn)
        def _wrap(clsorself, *args, **kwargs):
            try:
                ret = fcn(clsorself, *args, **kwargs)
            except NoEmission:
                return None

            return this.handle(*cls.__handle_args(lst, myrt, ret, args, kwargs))
        return _wrap

    @classmethod
    def __decorate_int(cls, lst, myrt, fcn):
        "returns a decorator for wrapping methods"
        myrt = EmitPolicy.get(myrt, fcn)

        @wraps(fcn)
        def _wrap(self, *args, **kwargs):
            try:
                ret = fcn(self, *args, **kwargs)
            except NoEmission:
                return None

            return self.handle(*cls.__handle_args(lst, myrt, ret, args, kwargs))
        return _wrap

    @classmethod
    def __decorate_func(cls, this, lst, myrt, fcn):
        "returns a decorator for wrapping free functions"
        myrt = EmitPolicy.get(myrt, fcn)

        @wraps(fcn)
        def _wrap(*args, **kwargs):
            try:
                ret = fcn(*args, **kwargs)
            except NoEmission:
                return None

            return this.handle(*cls.__handle_args(lst, myrt, ret, args, kwargs))
        return _wrap

    def __add_func(  # pylint: disable=too-many-arguments
            self,
            lst:       Iterable[str],
            fcn:       _OBSERVERS,
            decorate:  Optional[Callable[[_OBSERVERS], Callable]] = None,
            test:      Optional[Callable[..., bool]]              = None,
    ):
        if isinstance(fcn, cast(type, staticmethod)):
            fcn = getattr(fcn, '__func__')

        elif ismethod(fcn):
            raise NotImplementedError(
                "observe cannot decorate a method unless it's a static one"
            )

        elif not callable(fcn):
            raise ValueError("observer must be callable")

        if decorate is not None:
            fcn = decorate(fcn)

        if test is not None:
            @wraps(HashFunc.callable(fcn))
            def _fcn(*args, __fcn__ = fcn, __test__ = test, **kwargs):
                return __fcn__(*args, **kwargs) if __test__(*args, **kwargs) else None

            fcn = _fcn

        for name in lst:
            if self.__SIMPLE.match(name):
                cur = self._handlers.setdefault(name.lower().strip(), set())
                cast(Set, cur).add(fcn)
            else:
                dcur = self._handlers.setdefault('ㄡ', {})
                cast(_COMPLETIONS, dcur).setdefault(
                    cast(Callable, fcn), set()
                ).add(re.compile(name))

        return fcn

def _compare(cur, val):
    if type(cur) is not type(val):
        return False
    try:
        return cur == val
    except Exception:  # pylint: disable=broad-except
        return pickle.dumps(cur) == pickle.dumps(val)
    return False

class Controller(Event):
    "Main controller class"

    @classmethod
    def emit(cls, *names, returns = EmitPolicy.annotations):
        "decorator for emitting signals: can only be applied to *Controller* classes"
        return Event.internalemit(*names, returns = returns)

    @staticmethod
    def updatemodel(model, **kwargs) -> dict:
        "updates a model element"
        old = dict()
        for name, val in kwargs.items():
            cur = getattr(model, name)
            if _compare(cur, val):
                continue

            old[name] = getattr(model, name)
            setattr(model, name, val)

        if len(old) is None:
            raise NoEmission()

        return old
