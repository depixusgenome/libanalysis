#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methods for building modal dialogs
"""
from   typing    import List, Dict, Tuple, Optional
from   typing.io import TextIO # pylint: disable=import-error
import re

from   utils.logconfig import getLogger
LOGS   = getLogger(__name__)

class BodyParser:
    "Builds the modal dialog's body"
    _SPLIT  = re.compile(r"\s\s+")
    _ARG    = re.compile(r"%\(((?:[^[\].]*?|(?:\[\d+\])?|\.?)+)\)\.?(\d+)?([dfs])?")
    _STYLE  = re.compile(r"^(.*?)!([^!]*)!$")
    _ELEM   = re.compile(r"(\w+)(?:(\[\d+\]))?")
    _SEP    = '!height:20px;!'
    _ROWT   = '!font-style:italic;font-weight:bold;!'
    _COLT   = '!font-style:italic;font-weight:normal;!'
    @classmethod
    def tohtml(cls, body, model, default) -> Dict[str, Optional[str]]:
        "creates the dialog's body"
        tmp = [
            j.strip()
            for j in (
                body.strip().split("\n") if isinstance(body, str) else body
            )
        ]
        found = True
        body  = []
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
            body.append(i)

        try:
            out: Tuple[Optional[str], str] = cls.__tohtml(body, model, default)
        except Exception as exc: # pylint: disable=broad-except
            LOGS.exception(exc)
            out = "Error!", f"<p>{str(exc)}</p>"
        return dict(body = out[1], title = out[0])

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
    def __tohtml(cls, body, model, default) -> Tuple[Optional[str], str]:
        "creates the dialog's body"
        ntitles = sum(1 for i in body if i.startswith('# '))
        nsubs   = sum(1 for i in body if i.startswith('## '))
        if ntitles > 1 and nsubs > 1:
            raise RuntimeError("could not parse dialog")

        if ntitles+nsubs <= 1:
            return cls.__totab(body, model, default)

        if ntitles == 0:
            title = None

        titleflag = '^^!^:^' if ntitles >= 2 else "# "
        tabflag   = '# '     if ntitles >= 2 else "## "

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
        return title, cls.jointabs(tabs)

    @classmethod
    def jointabs(cls, tabs):
        "return html"
        return (
            (
                "<div class='dpx-span'>"
                +"".join(cls.__htmltitle(i, j[0],  0) for i, j in enumerate(tabs))
                +"</div>"
            )
            +"".join(cls.__htmlbody(i, j[1], 0)  for i, j in enumerate(tabs))
        )

    @staticmethod
    def __htmltitle(btn:int, title:str, ind: int) -> str:
        "return the html version of the title"
        fcn  = "Bokeh.DpxModal.prototype.clicktab"
        head = "cur" if btn == ind else ""
        return (
            "<button type='button'"
            +f" class='bk-bs-btn bk-bs-btn-default bbm-dpx-{head}btn'"
            +f" id='bbm-dpx-btn-{btn}'"
            +f' onclick="{fcn}({btn})">'
            +f'{title}</button>'
        )

    @staticmethod
    def __htmlbody(btn:int, body, ind:int) -> str:
        "return the html version of the body"
        head = 'curtab' if btn == ind else 'hidden'
        return f'<div class="bbm-dpx-{head}" id="bbm-dpx-tab-{btn}">{body}</div>'

    @classmethod
    def __parseargs(cls, model, default, body: List[str]) -> List[List[str]]:
        body    = [
            j if j else cls._SEP
            for j in body
            if not (j.startswith("# ") or j.startswith("## "))
        ]

        out     = [cls._SPLIT.split(j) for j in body]
        for lst in out:
            args = [(j, cls._ARG.match(j)) for j in lst]
            vals = [cls.__eval(k, model)   for j, k in args]
            dfl  = [cls.__eval(k, default) for j, k in args]
            if any(k != l for k, l in zip(vals, dfl)):
                lst[0] += f"({', '.join(dfl)})"
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
                    +''.join(f"<td {j[1]}>{j[0]}</td>" for j in styles[1:])
                    +'</tr>'
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

def tohtml(body, model, default) -> Dict[str, Optional[str]]:
    "return the title and the body for a modaldialog"
    return BodyParser.tohtml(body, model, default)

def changelog(stream:TextIO, appname:str):
    "extracts default startup message from a changelog"
    head = '<h2 id="'+appname.lower().split('_')[0].replace('app', '')
    line = ""
    for line in stream:
        if line.startswith(head):
            break
    else:
        return None

    newtab = lambda x: [x.split('>')[1].split('<')[0].split('_')[1], ""]
    tabs: List[List[str, str]] = [newtab(line)]  # type: ignore
    for line in stream:
        if line.startswith('<h2'):
            tabs.append(newtab(line))
        elif line.startswith('<h1'):
            break
        else:
            tabs[-1][-1] += line
    return BodyParser.jointabs(tabs)
