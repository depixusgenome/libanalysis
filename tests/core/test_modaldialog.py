#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the modal dialog
"""
from contextlib import contextmanager
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", ".* from 'collections'.*", DeprecationWarning)
    import modaldialog.options as opts

class _Dummy:
    def __init__(self, **kwa):
        self.__dict__.update(kwa)

def test_text():
    "test text option"

    mdl = _Dummy(
        first  = "first",
        second = ['second', 'ss'],
        third  = [_Dummy(third = "third")],
        fourth = "fourth",
        fifth  = "fifth",
        sixth  = "sixth"
    )
    txt = """
        1 %(first)s
        2 %(second[0])s
        3 %(third[0].third)s
        4 %(fourth)10s
        5 %(fifth{placeholder="5" class="dpx-5"})10s
        6 %(sixth{placeholder="5" style='heigh: 5px;'})10s
    """

    body  = [i.strip() for i in opts.tohtml(txt, mdl).strip().split("\n")]
    # pylint: disable=line-too-long
    truth = [
        """1 <input type="text" name="first"  value="first" class='bk-widget-form-input'>""",
        """2 <input type="text" name="second[0]"  value="second" class='bk-widget-form-input'>""",
        """3 <input type="text" name="third[0].third"  value="third" class='bk-widget-form-input'>""",
        """4 <input type="text" name="fourth"  value="fourth" style='width: 10px;' class='bk-widget-form-input'>""",
        """5 <input type="text" name="fifth"  value="fifth" placeholder="5" class="bk-widget-form-input dpx-5" style='width: 10px;'>""",
        """6 <input type="text" name="sixth"  value="sixth" placeholder="5" style='width: 10px; heigh: 5px;' class='bk-widget-form-input'>"""
    ]
    assert body == truth

    itms = {
        'first': 'aaa',
        'second[0]': 'bbb',
        'third[0].third': 'ccc',
        'fourth': 'ddd',
        'sixth': 'fff'
    }

    assert opts.fromhtml(itms, txt, mdl) is None
    # pylint: disable=no-member
    assert mdl.first == 'aaa'
    assert mdl.second == ['bbb', 'ss']
    assert mdl.third[0].third == 'ccc'
    assert mdl.fourth == 'ddd'
    assert mdl.fifth == 'fifth'
    assert mdl.sixth == 'fff'

def test_int():
    "test int option"

    mdl = _Dummy(
        first  = 1,
        second = [2,22],
        third  = [_Dummy(third = 3)],
        fourth = 4,
        fifth  = 5,
        sixth  = 6
    )
    txt = """
        1 %(first)i
        2 %(second[0])d
        3 %(third[0].third)D
        4 %(fourth)oi
        5 %(fifth{placeholder="5" class="dpx-5"})od
        6 %(sixth{placeholder="5" style='heigh: 5px;'})I
    """

    body  = [i.strip() for i in opts.tohtml(txt, mdl).strip().split("\n")]
    # pylint: disable=line-too-long
    truth = [
        """1 <input type="number" name="first"  value="1" class='bk-widget-form-input'>""",
        """2 <input type="number" name="second[0]"  value="2" class='bk-widget-form-input'>""",
        """3 <input type="number" name="third[0].third"  min=0 value="3" class='bk-widget-form-input'>""",
        """4 <input type="number" name="fourth"  value="4" class='bk-widget-form-input'>""",
        """5 <input type="number" name="fifth"  value="5" placeholder="5" class="bk-widget-form-input dpx-5">""",
        """6 <input type="number" name="sixth"  min=0 value="6" placeholder="5" style=\'heigh: 5px;\' class=\'bk-widget-form-input\'>"""
    ]
    assert body == truth

    itms = {
        'first': '11',
        'second[0]': '21',
        'third[0].third':'31',
        'fourth': '41',
        'sixth': '61',
    }

    out = []
    @contextmanager
    def _dumm(**kwa):
        out.append(kwa)
        yield

    assert opts.fromhtml(itms, txt, mdl, _dumm, xxx =1) is None
    assert out == [{'xxx': 1}]

    # pylint: disable=no-member
    assert mdl.first == 11
    assert mdl.second == [21, 22]
    assert mdl.third[0].third == 31
    assert mdl.fourth == 41
    assert mdl.fifth == 5
    assert mdl.sixth == 61

def test_float():
    "test int option"

    mdl = _Dummy(
        first  = 1,
        second = [2],
        third  = [_Dummy(third = 3)],
        fourth = 4,
        fifth  = 5,
        sixth  = 6
    )
    txt = """
        1 %(first)f
        2 %(second[0])f
        3 %(third[0].third)F
        4 %(fourth)of
        5 %(fifth{placeholder="5" class="dpx-5"}).3of
        6 %(sixth{placeholder="5" style='heigh: 5px;'})F
    """

    body  = [i.strip() for i in opts.tohtml(txt, mdl).strip().split("\n")]
    # pylint: disable=line-too-long
    truth = [
        """1 <input type="number" name="first"  value="1" class='bk-widget-form-input'>""",
        """2 <input type="number" name="second[0]"  value="2" class='bk-widget-form-input'>""",
        """3 <input type="number" name="third[0].third"  min=0 value="3" class='bk-widget-form-input'>""",
        """4 <input type="number" name="fourth"  value="4" class='bk-widget-form-input'>""",
        """5 <input type="number" name="fifth" step=0.001 value="5" placeholder="5" class="bk-widget-form-input dpx-5">""",
        """6 <input type="number" name="sixth"  min=0 value="6" placeholder="5" style=\'heigh: 5px;\' class=\'bk-widget-form-input\'>"""
    ]
    assert body == truth

    itms = {
        'first': '11',
        'second[0]': '21.111',
        'third[0].third':'31.111',
        'fourth': '',
        'sixth': '6e3',
    }

    assert opts.fromhtml(itms, txt, mdl) is None

    # pylint: disable=no-member
    assert mdl.first == 11
    assert mdl.second == [21.111]
    assert mdl.third[0].third == 31.111
    assert mdl.fourth is None
    assert mdl.fifth == 5
    assert mdl.sixth == 6e3

def test_csv():
    "test int option"

    mdl = _Dummy(
        first  = [],
        second = [[1,2]],
        third  = [_Dummy(third = ['a','b'])],
        fourth = [1.1, 2.2],
    )
    txt = """
        1 %(first)csv
        2 %(second[0])csvd
        3 %(third[0].third)csv
        4 %(fourth)csvf
    """

    body  = [i.strip() for i in opts.tohtml(txt, mdl).strip().split("\n")]
    # pylint: disable=line-too-long
    truth = [
        """1 <input type="text" name="first"  value = ""  placeholder="comma separated values"  class='bk-widget-form-input'>""",
        r"""2 <input type="text" name="second[0]"  value = "1, 2"  placeholder="comma separated integers"  pattern="[\d,;:]*"  title="comma separated integers"  class='bk-widget-form-input'>""",
        """3 <input type="text" name="third[0].third"  value = "a, b"  placeholder="comma separated values"  class='bk-widget-form-input'>""",
        r"""4 <input type="text" name="fourth"  value = "1.1, 2.2"  placeholder="comma separated floats"  pattern="[\d\.,;:]*"  title="comma separated floats"  class='bk-widget-form-input'>""",
    ]
    assert body == truth

    itms = {
        'first': 'aaa; bbb',
        'second[0]': '21',
        'third[0].third':'',
        'fourth': '41.111,',
    }

    assert opts.fromhtml(itms, txt, mdl) is None

    # pylint: disable=no-member
    assert mdl.first == ('aaa', ' bbb')
    assert mdl.second == [(21,)]
    assert mdl.third[0].third == ['a', 'b']
    assert mdl.fourth == (41.111,)

def test_check():
    "test int option"

    mdl = _Dummy(
        first  = True,
        second = [False],
        third  = [_Dummy(third = [False])],
    )
    txt = """
        1 %(first)b
        2 %(second[0])b
        3 %(third[0].third)b
    """

    body  = [i.strip() for i in opts.tohtml(txt, mdl).strip().split("\n")]
    # pylint: disable=line-too-long
    truth = [
        """1 <input type="checkbox" name="first" checked class='bk-bs-checkbox bk-widget-form-input'/>""",
        """2 <input type="checkbox" name="second[0]"  class='bk-bs-checkbox bk-widget-form-input'/>""",
        """3 <input type="checkbox" name="third[0].third" checked class='bk-bs-checkbox bk-widget-form-input'/>""",
    ]
    assert body == truth

    itms = {'first': False }
    assert opts.fromhtml(itms, txt, mdl) is None

    # pylint: disable=no-member
    assert mdl.first is False
    assert mdl.second == [False]

def test_choice(monkeypatch):
    "test int option"

    mdl = _Dummy(
        first  = "aaa",
        second = ["bbb"],
        third  = [_Dummy(third = "ccc")],
    )
    txt = """
        1 %(first)|aaa:choice1|bbb:choice2|ccc:choice3|
        2 %(second[0])|aaa:choice1|bbb:choice2|ccc:choice3|
        3 %(third[0].third)|aaa:choice1|bbb:choice2|ccc:choice3|
    """

    import random
    monkeypatch.setattr(random, 'randint', lambda *x:1111)
    body  = [i.strip() for i in opts.tohtml(txt, mdl).strip().split("\n")]

    # pylint: disable=line-too-long
    truth = [
        """1 <select name="first" id="first1111" ><option selected="selected" value="aaa">choice1</option><option value="bbb">choice2</option><option value="ccc">choice3</option></select>""",
        """2 <select name="second[0]" id="second[0]1111" ><option value="aaa">choice1</option><option selected="selected" value="bbb">choice2</option><option value="ccc">choice3</option></select>""",
        """3 <select name="third[0].third" id="third[0].third1111" ><option value="aaa">choice1</option><option value="bbb">choice2</option><option selected="selected" value="ccc">choice3</option></select>""",
    ]
    assert body == truth

    itms = {'first': 'bbb'}
    assert opts.fromhtml(itms, txt, mdl) is None

    # pylint: disable=no-member
    assert mdl.first  == 'bbb'
    assert mdl.second == ['bbb']

if __name__ == '__main__':
    test_csv()
