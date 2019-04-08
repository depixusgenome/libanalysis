#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optionos to be used by the modal dialog
"""
from    typing                  import Callable, ContextManager, Union, Dict, Any, cast
from    functools               import partial
from    abc                     import ABCMeta, abstractmethod
from    contextlib              import contextmanager
import  re
import  random
import  numpy                   as np

from    utils.logconfig         import getLogger
LOGS  = getLogger()
class Option(metaclass = ABCMeta):
    "Converts a text tag to an html input"
    NAME  = r'%\((?P<name>[\w\.\[\]]*)\s*(?:{\s*(?P<attr>[^{}]*?)?\s*}\s*)?\)'
    _cnv  = None

    @abstractmethod
    def replace(self, model, body:str) -> str:
        "replaces a pattern by an html tag"

    @abstractmethod
    def converter(self, model, body:str) -> Callable:
        "returns a method which sets values in a model"

    @classmethod
    def _addtoattr(cls, current, key, value):
        if not current:
            return f"{key}='{value}'"
        if key not in current:
            return f"{current} {key}='{value}'"
        return re.sub(
            fr'({key})\s*=\s*(["\'])',
            lambda x: f'{x.group(1)}={x.group(2)}{value} ',
            current
        )

    def _default_empty(self, elems, model, key):
        if elems[key]:
            self.setvalue(model, key, None)
        elif self._cnv is str:
            self.setvalue(model, key, '')

    def _default_apply(self, model, elems, # pylint: disable=too-many-arguments
                       cnv, storeempty, key, val):
        if key not in elems:
            return False

        if val != '':
            try:
                converted = cnv(val)
            except Exception as exc: # pylint: disable=broad-except
                LOGS.exception(exc)
            else:
                self.setvalue(model, key, converted)
        elif isinstance(storeempty, Exception):
            raise storeempty
        else:
            storeempty(model, key)
        return True

    def _converter(self, model, elems, cnv, storeempty = None) -> Callable:
        "returns a method which sets values in a model"
        if storeempty is None:
            storeempty = partial(self._default_empty, elems)
        fcn = partial(self._default_apply, model, elems, cnv, storeempty)
        return cast(Callable, fcn)

    _INDEX = re.compile(r"(\w+)\[(\d+)\]")
    def getvalue(self, mdl, keystr, default):
        "gets the value in the model"
        if isinstance(mdl, dict):
            return mdl[keystr]

        keys = keystr.split('.')
        for key in keys[:-1]:
            match = self._INDEX.match(key)
            if match:
                mdl = getattr(mdl, match.group(1))[int(match.group(2))]
            else:
                mdl = getattr(mdl, key)

        match = self._INDEX.match(keys[-1])
        if match:
            return getattr(mdl, match.group(1), default)[int(match.group(2))]
        return getattr(mdl, keys[-1], default)

    def setvalue(self, mdl, keystr, val):
        "sets the value in the model"
        if isinstance(mdl, dict):
            mdl[keystr] = val
        else:
            keys = keystr.split('.')
            for key in keys[:-1]:
                match = self._INDEX.match(key)
                if match:
                    mdl = getattr(mdl, match.group(1))[int(match.group(2))]
                else:
                    mdl = getattr(mdl, key)

            match = self._INDEX.match(keys[-1])
            if match:
                getattr(mdl, match.group(1))[int(match.group(2))] = val
            else:
                setattr(mdl, keys[-1], val)

class ChoiceOption(Option):
    "Converts a text tag to an html check"
    _PATT = re.compile(Option.NAME+r'(?P<cols>(?:\|\w+\:[^|{}<>]+)*)\|')
    def converter(self, model, body:str) -> Callable:
        "returns a method which sets values in a model"
        elems = frozenset(i.group('name') for i in self._PATT.finditer(body))
        return self._converter(model, elems, lambda x: x, AssertionError())

    def replace(self, model, body:str) -> str:
        "replaces a pattern by an html tag"
        def _replace(match):
            key   = match.group('name')
            attr  = match.group('attr') or ''
            ident = key+str(random.randint(0,100000))
            out   = '<select name="{}" id="{}" {}>'.format(key, ident, attr)
            val   = ''
            for i in match.group('cols')[1:].split("|"):
                val = self.getvalue(model, key, i.split(":")[0])
                break

            for i in match.group('cols')[1:].split("|"):
                i    = i.split(':')
                sel  = 'selected="selected" ' if i[0] == str(val) else ""
                out += '<option {}value="{}">{}</option>'.format(sel, *i)
            return out.format(ident)+'</select>'
        return self._PATT.sub(_replace, body)

class CheckOption(Option):
    "Converts a text tag to an html check"
    _PATT = re.compile(Option.NAME+'b')
    @staticmethod
    def __cnv(val):
        if val in ('on', True):
            return True
        if val in ('off', False):
            return False
        raise ValueError()

    def converter(self, model, body:str) -> Callable:
        "returns a method which sets values in a model"
        elems = frozenset(i.group('name') for i in self._PATT.finditer(body))
        return self._converter(model, elems, self.__cnv, AssertionError())

    def replace(self, model, body:str) -> str:
        "replaces a pattern by an html tag"
        def _replace(match):
            key  = match.group('name')
            attr = self._addtoattr(
                match.group("attr"),
                "class",
                "bk-bs-checkbox bk-widget-form-input"
            )
            assert len(key), "keys must have a name"
            val = 'checked' if bool(self.getvalue(model, key, False)) else ''
            return '<input type="checkbox" name="{}" {} {}/>'.format(key, val, attr)

        return self._PATT.sub(_replace, body)

class TextOption(Option):
    "Converts a text tag to an html text input"
    def __init__(self, cnv, patt, step):
        self._cnv  = cnv
        self._patt = re.compile(self.NAME+patt)
        self._step = step

    def converter(self, model, body:str) -> Callable:
        "returns a method which sets values in a model"
        elems = {i.group('name'): i.group('opt') == 'o' for i in self._patt.finditer(body)}
        return self._converter(model, elems, self._cnv)

    def replace(self, model, body:str) -> str:
        "replaces a pattern by an html tag"
        def _replace(info, tpe):
            key = info['name']
            assert len(key), "keys must have a name"

            opt = self.__step(info)
            if info.get("fmt", "s").upper() == info.get("fmt", "s"):
                opt += " min=0"

            opt += self.__value(model, key, info)

            attr  = info.get('attr', '') or ''
            if info.get('width', None):
                attr = self._addtoattr(attr, "style", f'width: {info["width"]}px;')
            attr = self._addtoattr(attr, "class", 'bk-widget-form-input')

            inpt = '<input type="{}" name="{}" {} {}>'
            return inpt.format(tpe, key, opt, attr)

        tpe = 'text' if self._cnv is str else 'number'
        fcn = lambda i: _replace(i.groupdict(), tpe)
        return self._patt.sub(fcn, body)

    def __step(self, info) -> str:
        return (
            ''                        if self._step in (0, None)     else
            'step='+str(self._step)   if isinstance(self._step, int) else
            ''                        if info[self._step] is None    else
            'step=0.'+'0'*(int(info[self._step])-1)+'1'
        )

    def __value(self, model, key, info) -> str:
        val = self.getvalue(model, key, None)
        if val in (None, ""):
            return ""

        if isinstance(self._step, int):
            val = np.around(val, int(self._step))
        elif info.get(self._step, None) is not None:
            val = np.around(val, int(info[self._step]))
        return ' value="{}"'.format(val)

class CSVOption(Option):
    "Converts a text tag to an html text input"
    def __init__(self, cnv, patt):
        super().__init__()
        split      = re.compile('[,;:]').split
        self._cnv  = lambda i: tuple(cnv(j) for j in split(i) if j)
        self._patt = re.compile(self.NAME+patt)
        self._mask = (
            r'[\d\.,;:]*'   if patt[-1] == 'f' else
            r'[\d,;:]*'     if patt[-2] == 'd' else
            ""
        )
        self._title = (
            'floats'   if patt[-1] == 'f' else
            'integers' if patt[-2] == 'd' else
            'values'
        )

    def converter(self, model, body:str) -> Callable:
        "returns a method which sets values in a model"
        elems = {i.group('name'): i.group('opt') == 'o' for i in self._patt.finditer(body)}
        return self._converter(model, elems, self._cnv)

    def replace(self, model, body:str) -> str:
        "replaces a pattern by an html tag"
        def _replace(match):
            key = match.group('name')
            assert len(key), "keys must have a name"
            attr = match.group("attr") or ''

            val  = self.getvalue(model, key, None)
            opt  = f""" value = "{', '.join(str(i) for i in val) if val else ''}" """
            if "placeholder" not in attr:
                opt += f' placeholder="comma separated {self._title}" '
            if self._mask and "pattern" not in attr:
                opt += f' pattern="{self._mask}" '
                if "title" not in attr:
                    opt += f' title="comma separated {self._title}" '

            attr = self._addtoattr(attr, "class", 'bk-widget-form-input')
            if match.group('width'):
                attr = self._addtoattr(attr, "style", f'width: {match.group("width")}px;')

            inpt = '<input type="text" name="{}" {} {}>'
            return inpt.format(key, opt, attr)

        return self._patt.sub(_replace, body)

_PREC   = r'(?:\.(?P<prec>\d*))?'
_OPT    = r'(?P<opt>o)?'
OPTIONS = (CheckOption(),
           TextOption(int,   _OPT+r'(?P<fmt>[idID])',     0),
           TextOption(float, _PREC+_OPT+r'(?P<fmt>[fF])', 'prec'),
           TextOption(str,   _OPT+r'(?P<width>\d*)s',    None),
           CSVOption(int,    _OPT+r'(?P<width>\d*)csv[id]'),
           CSVOption(float,  _OPT+r'(?P<width>\d*)csvf'),
           CSVOption(str,    _OPT+r'(?P<width>\d*)csv'),
           ChoiceOption())

def _build_elem(val):
    if isinstance(val, tuple):
        return f'<td style="{val[0]}">'+val[1]+'</td>'
    return f'<td>'+val+'</td>'

@contextmanager
def _dummy():
    yield

def tohtml(body, model) -> str:
    "convert to html"
    if isinstance(body, (tuple, list)):
        if len(body) == 0:
            return ""

        if hasattr(body[0], 'tohtml'):
            body = body[0].tohtml(body)
        else:
            body = '<table>' + (''.join('<tr>'
                                        + ''.join(_build_elem(i) for i in j)
                                        + '</tr>'
                                        for j in body)) + '</table>'

    for tpe in OPTIONS:
        body = tpe.replace(model, body)
    return body

def fromhtml(
        itms: Dict[str, Any],
        body,
        model,
        context: Union[None, Callable, ContextManager] = None,
        **kwa
):
    "extract changes from the html"
    if isinstance(body, (list, tuple)):
        if len(body) and hasattr(body[0], 'body'):
            body = sum((tuple(i.body) for i in body), ())
        body = ' '.join(
            ' '.join(k if isinstance(k, str) else k[1] for k in i)
            for i in body
        )

    converters = [i.converter(model, body) for i in OPTIONS]
    ordered    = sorted(itms.items(), key = lambda i: body.index('%('+i[0]))
    if context is None:
        for i in ordered:
            any(cnv(*i) for cnv in converters)
    elif isinstance(context, ContextManager):
        with context:
            for i in ordered:
                any(cnv(*i) for cnv in converters)
    else:
        with context(**kwa):
            for i in ordered:
                any(cnv(*i) for cnv in converters)