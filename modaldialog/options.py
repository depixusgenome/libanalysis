#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options to be used by the modal dialog
"""
from    typing                  import (
    Callable, ContextManager, Union, Dict, Any, Tuple, Iterable, Optional, cast
)
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

    @classmethod
    def _default_empty(cls, elems, cnv, model, key):
        if elems[key]:
            cls.setvalue(model, key, None)
        elif cnv is str:
            cls.setvalue(model, key, '')

    @classmethod
    def _default_apply(  # pylint: disable=too-many-arguments
            cls, model, elems, cnv, storeempty, key, val
    ):
        if key not in elems:
            return False

        if val != '':
            try:
                converted = cnv(val)
            except Exception as exc:  # pylint: disable=broad-except
                LOGS.exception(exc)
            else:
                cls.setvalue(model, key, converted)
        elif isinstance(storeempty, Exception):
            raise storeempty
        else:
            storeempty(model, key)
        return True

    @classmethod
    def _converter(cls, model, elems, cnv, storeempty = None) -> Callable:
        "returns a method which sets values in a model"
        if storeempty is None:
            storeempty = partial(cls._default_empty, elems, cnv)
        fcn = partial(cls._default_apply, model, elems, cnv, storeempty)
        return cast(Callable, fcn)

    _INDEX = re.compile(r"(\w+)\[(\d+)\]")

    @classmethod
    def getvalue(cls, mdl, keystr, default):
        "gets the value in the model"
        if isinstance(mdl, dict):
            return mdl[keystr]

        keys = keystr.split('.')
        for key in keys[:-1]:
            match = cls._INDEX.match(key)
            if match:
                mdl = getattr(mdl, match.group(1))[int(match.group(2))]
            else:
                mdl = getattr(mdl, key)

        match = cls._INDEX.match(keys[-1])
        if match:
            return getattr(mdl, match.group(1), default)[int(match.group(2))]
        return getattr(mdl, keys[-1], default)

    @classmethod
    def setvalue(cls, mdl, keystr, val):
        "sets the value in the model"
        if isinstance(mdl, dict):
            mdl[keystr] = val
        else:
            keys     = keystr.split('.')
            old, mdl = cls.__getancestors(mdl, keys)
            if cls.__setsequence(old, mdl, keys, val):
                if cls.__setnamedtuple(old, mdl, keys, val):
                    cls.__setattr(mdl, keys, val)

    @classmethod
    def __getancestors(cls, mdl, keys) -> Tuple[Any, Any]:
        root = old = mdl
        try:
            for key in keys[:-1]:
                match = cls._INDEX.match(key)
                old   = mdl
                if match:
                    mdl = getattr(mdl, match.group(1))[int(match.group(2))]
                else:
                    mdl = getattr(mdl, key)
        except AttributeError as exc:
            raise AttributeError(f"Can't get {root}.{'.'.join(keys[-1])}") from exc
        return old, mdl

    @classmethod
    def __setsequence(cls, old, mdl, keys, val) -> bool:
        match    = cls._INDEX.match(keys[-1])
        if match is None:
            return True
        try:
            ind = int(match.group(2))

            if isinstance(getattr(mdl, match.group(1)), tuple):
                old, mdl = mdl, getattr(mdl, match.group(1))
                val      = type(mdl)(val if i == ind else mdl[i] for i in range(len(mdl)))
                setattr(old, match.group(1), val)

            elif isinstance(mdl, tuple):
                val = type(mdl)(val if i == ind else mdl[i] for i in range(len(mdl)))
                setattr(old, keys[-2], val)
            else:
                getattr(mdl, match.group(1))[ind] = val
        except AttributeError as exc:
            raise AttributeError(f"Can't set {mdl}{keys[-1]} = {val}") from exc
        except TypeError as exc:
            raise TypeError(f"Can't set {mdl}{keys[-1]} = {val}") from exc
        return False

    @classmethod
    def __setnamedtuple(cls, old, mdl, keys, val) -> bool:
        if not isinstance(mdl, tuple):
            return True
        try:
            ind = getattr(type(mdl), '_fields').index(keys[-1])
            val = type(mdl)(*(val if i == ind else mdl[i] for i in range(len(mdl))))
            setattr(old, keys[-2], val)
        except AttributeError as exc:
            raise AttributeError(f"Can't set {old}.{keys[-2]} = {val}") from exc
        return False

    @classmethod
    def __setattr(cls, mdl, keys, val):
        try:
            setattr(mdl, keys[-1], val)
        except AttributeError as exc:
            raise AttributeError(f"Can't set {mdl}.{keys[-1]} = {val}") from exc


class ChoiceOption(Option):
    "Converts a text tag to an html check"
    _PATT = re.compile(Option.NAME+r'(?P<cols>(?:\|\w+\:[^|{}<>]+)*)\|')
    @classmethod
    def converter(cls, model, body:str) -> Callable:
        "returns a method which sets values in a model"
        elems = frozenset(i.group('name') for i in cls._PATT.finditer(body))
        return cls._converter(model, elems, lambda x: x, AssertionError())

    @classmethod
    def replace(cls, model, body:str) -> str:
        "replaces a pattern by an html tag"
        def _replace(match):
            key   = match.group('name')
            attr  = match.group('attr') or ''
            ident = key+str(random.randint(0,100000))
            val   = ''
            try:
                for i in match.group('cols')[1:].split("|"):
                    val = cls.getvalue(model, key, i.split(":")[0])
                    break
            except Exception as exc:   # pylint: disable=broad-except
                LOGS.exception(exc)
                attr += " disabled='true'"

            out = '<select name="{}" id="{}" {}>'.format(key, ident, attr)
            for i in match.group('cols')[1:].split("|"):
                i    = i.split(':')
                sel  = 'selected="selected" ' if i[0] == str(val) else ""
                out += '<option {}value="{}">{}</option>'.format(sel, *i)
            return out.format(ident)+'</select>'
        return cls._PATT.sub(_replace, body)


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

    @classmethod
    def converter(cls, model, body:str) -> Callable:
        "returns a method which sets values in a model"
        elems = frozenset(i.group('name') for i in cls._PATT.finditer(body))
        return cls._converter(model, elems, cls.__cnv, AssertionError())

    @classmethod
    def replace(cls, model, body:str) -> str:
        "replaces a pattern by an html tag"
        def _replace(match):
            key  = match.group('name')
            attr = cls._addtoattr(
                match.group("attr"),
                "class",
                "bk bk-input"
            )
            assert len(key), "keys must have a name"
            val = ''
            try:
                if bool(cls.getvalue(model, key, False)):
                    val = "checked"
            except Exception as exc:  # pylint: disable=broad-except
                LOGS.exception(exc)
                attr += " disabled='true'"
            return (
                '<div class ="bk bk-input-group">'
                + '<input type="checkbox" name="{}" {} {}/>'.format(key, val, attr)
                + "</div>"
            )

        return cls._PATT.sub(_replace, body)


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

            attr  = info.get('attr', '') or ''
            try:
                opt += self.__value(model, key, info)
            except Exception as exc:  # pylint: disable=broad-except
                LOGS.exception(exc)
                attr += " disabled='true'"

            if info.get('width', None):
                attr = self._addtoattr(attr, "style", f'min-width: {info["width"]}px;')
            attr = self._addtoattr(attr, "class", 'bk bk-input')

            inpt = '<input type="{}" name="{}" {} {}>'
            return inpt.format(tpe, key, opt, attr)

        tpe = 'text' if self._cnv is str else 'number'
        return self._patt.sub((lambda i: _replace(i.groupdict(), tpe)), body)

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
            r'[\d\.,;:\s]*'   if patt[-1] == 'f' else
            r'[\d,;:\s]*'     if patt[-2] == 'd' else
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

            val  = None
            try:
                val  = self.getvalue(model, key, None)
            except Exception as exc:  # pylint: disable=broad-except
                LOGS.exception(exc)
                attr += " disabled='true'"

            opt  = f""" value = "{', '.join(str(i) for i in val) if val else ''}" """
            if "placeholder" not in attr:
                opt += f' placeholder="comma separated {self._title}" '
            if self._mask and "pattern" not in attr:
                opt += f' pattern="{self._mask}" '
                if "title" not in attr:
                    opt += f' title="comma separated {self._title}" '

            attr = self._addtoattr(attr, "class", 'bk bk-input')
            if match.group('width'):
                attr = self._addtoattr(attr, "style", f'min-width: {match.group("width")}px;')

            inpt = '<div class="bk bk-input-group"><input type="text" name="{}" {} {}></div>'
            return inpt.format(key, opt, attr)

        return self._patt.sub(_replace, body)


class TabOption(Option):
    "Converts a text tag to an html check"
    _PATT = re.compile(
        r"(?:^|\n)\s*(?P<title>.*?)\[(?P<key>.*?)\s*:\s*(?P<val>[^\]]*)\]?\s*(?:!(?P<tags>[^!]*)!)?"
    )
    _FIND = re.compile(r'tabkey="(?P<key>[^"]*)"')
    @classmethod
    def match(
            cls, title:Optional[str]
    ) -> Union[Tuple[Optional[str], None, None, None], Tuple[str, str, str, str]]:
        "return the html version of the title"
        if title:
            match = cls._PATT.match(title)
            if match is not None:
                return cast(
                    Tuple[str, str, str, str],
                    tuple(map(match.group, ('title', 'key', 'val', 'tags')))
                )
        return (title, None, None, None)

    def converter(self, model, body:str) -> Callable:
        "returns a method which sets values in a model"
        elems = frozenset(i.group('key') for i in self._FIND.finditer(body))
        return self._converter(model, elems, lambda x: x, AssertionError())

    def replace(self, model, body:str) -> str:
        "replaces a pattern by an html tag"
        return body


_PREC   = r'(?:\.(?P<prec>\d*))?'
_OPT    = r'(?P<opt>o)?'

def _int(val: str) -> int:
    return int(float(val)) if '.' in val else int(val)


OPTIONS = (
    CheckOption(),
    TextOption(_int,  _OPT+r'(?P<fmt>[idID])',     0),
    TextOption(float, _PREC+_OPT+r'(?P<fmt>[fF])', 'prec'),
    TextOption(str,   _OPT+r'(?P<width>\d*)s',    None),
    CSVOption(_int,   _OPT+r'(?P<width>\d*)csv[id]'),
    CSVOption(float,  _OPT+r'(?P<width>\d*)csvf'),
    CSVOption(str,    _OPT+r'(?P<width>\d*)csv'),
    ChoiceOption(),
    TabOption()
)


def _build_elem(val):
    if isinstance(val, tuple):
        return f'<td style="{val[0]}">'+val[1]+'</td>'
    return f'<td>'+val+'</td>'


@contextmanager
def _dummy():
    yield


def tohtml(body: Union[str, Iterable], model: Any) -> str:
    """
    Convert to html, parsing the lines for inputs
    """
    strbody: str = ""
    if isinstance(body, str):
        strbody = cast(str, body)
    else:
        lines = list(body)
        if len(lines) > 0:
            if hasattr(lines[0], 'tohtml'):
                strbody = lines[0].tohtml(lines)
            else:
                strbody = (
                    '<table>'
                    + ''.join(
                        f"<tr>{''.join(_build_elem(i) for i in j)}</tr>"
                        for j in body
                    )
                    + '</table>'
                )

    if strbody:
        for tpe in OPTIONS:
            strbody = tpe.replace(model, strbody)
    return strbody


def fromhtml(
        itms: Dict[str, Any],
        body,
        model,
        context: Union[None, Callable, ContextManager] = None,
        **kwa
):
    "extract changes from the html"
    try:
        if isinstance(body, (list, tuple)):
            if len(body) and hasattr(body[0], 'body'):
                body = sum((tuple(i.body) for i in body), ())
            body = ' '.join(
                ' '.join(k if isinstance(k, str) else k[1] for k in i)
                for i in body
            )

        converters = [i.converter(model, body) for i in OPTIONS]
        ordered    = sorted(itms.items(), key = lambda i: body.find('%('+i[0]))
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
    except Exception as exc:  # pylint: disable=broad-except
        LOGS.exception(exc)
        raise
