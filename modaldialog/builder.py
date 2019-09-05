#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods for building modal dialogs
"""
from   typing    import List, Dict, Tuple, Optional, Iterable, TypeVar, Union, cast
from   typing.io import TextIO  # pylint: disable=import-error
import re

from   utils.logconfig import getLogger
from   .options        import TabOption
LOGS   = getLogger(__name__)

ModelType = TypeVar('ModelType')

def tohtml(
        body:    Union[str, Iterable[str]],
        model:   Optional[ModelType] = None,
        default: Optional[ModelType] = None
) -> Dict[str, Optional[str]]:
    """
    Creates the dialog's body.

    Parameters
    ----------
    body:
        A string or sequence of strings containting the model to create.
    model:
        The instance affected byt the dialog
    default:
        An instance, normaly of the same type as `model` but with all its
        attributes at their default value. If an attribute is not at it's
        default value, the latter is displayed automatically.

    Returns
    -------
    A dictionnary with the title and the body to use in the modal dialog

    Example
    -------

    A complex example with multiple tabs
    ```python
    html = tohtml(
        \"\"\"
        # Main title

        ## First tab: text

        First entry is a sub-attribute              %(first.attr)s
        Second entry using an element from a list   %(second[0])s
        Third has specific input html tags          %(third{placeholder="5" class="dpx-5"})s

        ## Second tab: ints (d) or floats (f)

        First entry accepts ints                    %(ints.first)d
        Second entry accepts floats                 %(floats.first)f
        Third entry accepts positive floats         %(floats.sec)F
        Fourth entry accepts ints or nothing        %(ints.third)od

        ## Second tab: others

        add a checkbox            %(others.first)b
        add a set of choices      %(choices)|key1:label1|key2:label2|key3:label3|
        \"\"\",
        model,
        default,
    )
    ```

    A simpler one used for displaying or hiding a bokeh glyph
    ```python
    html = tohtml(
        \"\"\"
        ## Configuration

        Display orphaned fringes     %(visible)b
        \"\"\",
        glyph
    )
    ```
    """
    return BodyParser.tohtml(body, model, default)

class BodyParser:
    """
    Builds the modal dialog's body
    """
    _SPLIT  = re.compile(r"\s\s+")
    _ARG    = re.compile(r"%\(((?:[^[\].]*?|(?:\[\d+\])?|\.?)+)\)\.?(\d+)?([dfs])?")
    _STYLE  = re.compile(r"^(.*?)!([^!]*)!$")
    _ELEM   = re.compile(r"(\w+)(?:(\[\d+\]))?")
    _SEP    = '!height:20px;!'
    _ROWT   = '!font-style:italic;font-weight:bold;!'
    _COLT   = '!font-style:italic;font-weight:normal;!'
    @classmethod
    def tohtml(
            cls,
            body:    Union[str, Iterable[str]],
            model:   Optional[ModelType] = None,
            default: Optional[ModelType] = None
    ) -> Dict[str, Optional[str]]:
        "Creates the dialog's body. see module method for documentation"
        if default is None:
            default = model
        tmp: Iterable[str] = (
            j.strip()
            for j in (body.strip().splitlines() if isinstance(body, str) else body)
        )
        found: bool      = True
        lines: List[str] = []
        for i in tmp:
            if found and not i:
                continue
            elif i.startswith("### "):
                found = False
                elems = cls._SPLIT.split(i)
                i     = elems[0][4:]+cls._ROWT+"  "+'  '.join(j+cls._COLT for j in elems[1:])
            elif i.startswith("#"):
                found = True
            elif not (found or i):
                i     = cls._SEP
                found = True
            else:
                found = False
            lines.append(i)

        try:
            out: Tuple[Optional[str], str] = cls.__tohtml(lines, model, default)
        except Exception as exc:  # pylint: disable=broad-except
            LOGS.exception(exc)
            out = "Error!", f"<p>{str(exc)}</p>"
        return dict(body = out[1], title = out[0])

    @classmethod
    def jointabs(cls, tabs: Iterable[Tuple[Optional[str], str]], cur: int = 0) -> str:
        """
        Create the HTML string for multiple tabs

        Parameters
        ----------
        tabs:
            The list of tabs. Each item is a pair of strings, title and body.
        cur:
            The index of the current tab.

        Returns
        -------
        The HTML string for all tabs
        """
        head = (
            "<div class='bk bk-btn-group'>"
            if any(TabOption.match(i)[-2] for i,_ in tabs) else
            "<div>"
        )
        return (
            (
                head
                + "".join(cls.__jointab_title(i, j[0],  cur) for i, j in enumerate(tabs))
                + "</div>"
            )
            + "".join(cls.__jointab_body(i, j[1], cur)  for i, j in enumerate(tabs))
        )

    @staticmethod
    def __title(body) -> Optional[str]:
        title = next((j[2:] for j in body if j.startswith("# ")), None)
        if title is None:
            return next((j[3:] for j in body if j.startswith("## ")), None)
        return title

    @classmethod
    def __totab(cls, body, model, default) -> Tuple[Optional[str], str]:
        out  = cls.__parseargs(model, default, body)
        html = cls.__table(out)
        return cls.__title(body), html

    @classmethod
    def __tohtml(
            cls,
            body:    List[str],
            model:   ModelType,
            default: ModelType
    ) -> Tuple[Optional[str], str]:
        "creates the dialog's body"
        ntitles: int = sum(1 for i in body if i.startswith('# '))
        nsubs:   int = sum(1 for i in body if i.startswith('## '))
        if ntitles > 1 and nsubs > 1:
            raise RuntimeError("could not parse dialog")

        if ntitles+nsubs <= 1:
            return cls.__totab(body, model, default)

        if ntitles == 0:
            title = None

        titleflag: str = '^^!^:^' if ntitles >= 2 else "# "
        tabflag:   str = '# '     if ntitles >= 2 else "## "

        tabs:  List[Tuple[Optional[str], str]] = []
        lines: List[str]                       = []

        for i in body:
            if i.startswith(titleflag):
                continue
            elif i.startswith(tabflag) and lines:
                if lines:
                    tabs.append(cls.__totab(lines, model, default))
                lines.clear()
            lines.append(i)

        if lines:
            tabs.append(cls.__totab(lines, model, default))

        title = cls.__title(body) if ntitles == 1 else None
        return title, cls.jointabs(tabs, cls.__getcurrenttab(tabs, model))

    @classmethod
    def __jointab_title(cls, btn:int, title:Optional[str], ind: int) -> str:
        "return the html version of the title"
        title, key, val, tags = TabOption.match(title)
        head                  = "curbtn bk-active" if btn == ind else "btn"
        return (
            "<button type='button'"
            + (f' tabkey="{key}" tabvalue="{val}" ' if key else " tabvalue='-'")
            + (f' {tags} ' if tags else "")
            + f" class='bk bk-btn bk-btn-default bbm-dpx-{head}'"
            + f" id='bbm-dpx-btn-{btn}'"
            + f' onclick="Bokeh.DpxModal.prototype.clicktab({btn})">'
            + f'{title if title else "Page "+str(btn)}</button>'
        )

    @staticmethod
    def __jointab_body(btn:int, body, ind:int) -> str:
        "return the html version of the body"
        head = 'curtab' if btn == ind else 'hidden'
        return f'<div class="bbm-dpx-{head}" id="bbm-dpx-tab-{btn}">{body}</div>'

    _TITLEKV = re.compile(r"^(?P<title>.*?)\[(?P<key>.*?)\s*:\s*(?P<val>.*?)\]")
    _NONE    = type('_NONE', (), {})
    @classmethod
    def __getcurrenttab(cls, tabs: List[Tuple[Optional[str], str]], model: ModelType) -> int:
        kvmatch = ((i, cls._TITLEKV.match(cast(str, j[0]))) for i, j in enumerate(tabs) if j[0])
        kvlist  = ((i, map(j.group, ('key', 'val'))) for i, j in kvmatch if j)
        for i, (j, k) in kvlist:
            val = TabOption.getvalue(model, j, cls._NONE)
            if val is cls._NONE or k != str(val):
                continue
            return i
        return 0

    @classmethod
    def __parseargs(cls, model, default, body: List[str]) -> List[List[str]]:
        body    = [
            j if j else cls._SEP
            for j in body
            if not (j.startswith("# ") or j.startswith("## "))
        ]

        out     = [cls._SPLIT.split(j) for j in body]
        for lst in out:
            args = (cls._ARG.match(j) for j in lst if '%(' in j)
            vals = [(cls.__eval(k, model), cls.__eval(k, default)) for k in args]
            if any(k != l for k, l in vals):
                lst[0] += f" ({', '.join(f'<u>{l}</u>' if k != l else f'{l}' for k, l in vals)})"
        return out

    @classmethod
    def __table(cls, out: List[List[str]]) -> str:
        tables: List[List[List[str]]] = [[]]
        for i in out:
            if i == [cls._SEP]:
                tables.append([])
            tables[-1].append(i)

        txt = []
        for table in tables:
            maxv = max((len(i) for i in table), default = 0)
            txt.append("<table>")
            for i in table:
                if len(i) == 0 or i == ['']:
                    continue

                styles = [
                    (
                        j if not k else k.group(1),
                        '' if not k else f'style="{k.group(2)}"'
                    )
                    for j, k in ((j, cls._STYLE.match(j)) for j in i)
                ]
                txt.append(
                    f"<tr {styles[0][1]}><td>{styles[0][0]}</td>{'<td></td>' * (maxv-len(i))}"
                    + ''.join(f"<td {j[1]}>{j[0]}</td>" for j in styles[1:])
                    + '</tr>'
                )
            txt.append("</table>")
        return ''.join(txt)

    @classmethod
    def __eval(cls, match, model):
        if match is None:
            return None
        name = match.group(1)
        if '{' in name:
            name = name[:name.find('{')]
        for i in cls._ELEM.split(name):
            if i not in (None, '', '.'):
                model = model[int(i[1:-1])] if i.startswith('[') else getattr(model, i)
        return model

def changelog(stream:TextIO, appname:str, docpath: Optional[str] = None) -> Optional[str]:
    "extracts default startup message from a changelog"
    head = '<h2 id="'+appname.lower().split('_')[0].replace('app', '')

    line = ""
    for line in stream:
        if line.startswith(head):
            break
    else:
        return None

    def _newtab(txt: str) -> List[str]:
        return [txt.split('>')[1].split('<')[0].split('_')[1], ""]

    tabs: List[List[str]] = [_newtab(line)]  # type: ignore
    for line in stream:
        if line.startswith('<h2'):
            tabs.append(_newtab(line))
        elif line.startswith('<h1'):
            break
        else:
            tabs[-1][-1] += line

    out = BodyParser.jointabs(cast(Iterable[Tuple[Optional[str], str]], tabs))
    if docpath is not None:
        return f"""
            <style>
                a:link, a:visited {{
                  background-color: green;
                  color: white;
                  padding: 15px 25px;
                  text-align: center;
                  text-decoration: none;
                  display: inline-block;
                }}

                a:hover, a:active {{ background-color: darkgreen;}}
            </style>
            <a href="{docpath}/{appname}/{appname}.html" target="_blank" style="float:right;">
                Read the doc!!!
            </a>
            {out}
            """
    return out
