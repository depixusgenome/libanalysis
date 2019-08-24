#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa: E501
"""
Test the modal dialog
"""
from contextlib import contextmanager
from io         import StringIO
import re
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", ".* from 'collections'.*", DeprecationWarning)
    import modaldialog.options as opts
    import modaldialog.builder as build

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
        """1 <input type="text" name="first"  value="first" class=\'bk bk-input\'>""",
        """2 <input type="text" name="second[0]"  value="second" class=\'bk bk-input\'>""",
        """3 <input type="text" name="third[0].third"  value="third" class=\'bk bk-input\'>""",
        (
            """4 <input type="text" name="fourth"  value="fourth" """
            """style=\'min-width: 10px;\' class=\'bk bk-input\'>"""
        ),
        (
            """5 <input type="text" name="fifth"  value="fifth" placeholder="5" """
            """class="bk bk-input dpx-5" style=\'min-width: 10px;\'>"""
        ),
        (
            """6 <input type="text" name="sixth"  value="sixth" placeholder="5" """
            """style=\'min-width: 10px; heigh: 5px;\' class=\'bk bk-input\'>"""
        )
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
        """1 <input type="number" name="first"  value="1" class=\'bk bk-input\'>""",
        """2 <input type="number" name="second[0]"  value="2" class=\'bk bk-input\'>""",
        """3 <input type="number" name="third[0].third"  min=0 value="3" class=\'bk bk-input\'>""",
        """4 <input type="number" name="fourth"  value="4" class=\'bk bk-input\'>""",
        (
            """5 <input type="number" name="fifth"  value="5" placeholder="5" """
            """class="bk bk-input dpx-5">"""
        ),
        (
            """6 <input type="number" name="sixth"  min=0 value="6" placeholder="5" """
            """style=\'heigh: 5px;\' class=\'bk bk-input\'>"""
        )
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
        second = (2,),
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
        """1 <input type="number" name="first"  value="1" class=\'bk bk-input\'>""",
        """2 <input type="number" name="second[0]"  value="2" class=\'bk bk-input\'>""",
        """3 <input type="number" name="third[0].third"  min=0 value="3" class=\'bk bk-input\'>""",
        """4 <input type="number" name="fourth"  value="4" class=\'bk bk-input\'>""",
        (
            """5 <input type="number" name="fifth" step=0.001 value="5" placeholder="5" """
            """class="bk bk-input dpx-5">"""
        ),
        (
            """6 <input type="number" name="sixth"  min=0 value="6" placeholder="5" """
            """style=\'heigh: 5px;\' class=\'bk bk-input\'>"""
        )
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
    assert mdl.second == (21.111,)
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
        (
            """1 <div class="bk bk-input-group"><input type="text" name="first"  """
            """value = ""  placeholder="comma separated values"  """
            """class=\'bk bk-input\'></div>"""
        ),
        (
            """2 <div class="bk bk-input-group"><input type="text" name="second[0]"  """
            """value = "1, 2"  placeholder="comma separated integers"  """
            r"""pattern="[\d,;:\s]*"  title="comma separated integers"  """
            """class=\'bk bk-input\'></div>"""
        ),
        (
            """3 <div class="bk bk-input-group"><input type="text" """
            """name="third[0].third"  value = "a, b"  """
            """placeholder="comma separated values"  class='bk bk-input'></div>"""
        ),
        (
            """4 <div class="bk bk-input-group"><input type="text" name="fourth"  """
            """value = "1.1, 2.2"  placeholder="comma separated floats"  """
            """pattern="[\\d\\.,;:\\s]*"  title="comma separated floats"  """
            """class='bk bk-input'></div>"""
        )

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
        (
            """1 <div class ="bk bk-input-group"><input type="checkbox" name="first" """
            """checked class=\'bk bk-input\'/></div>"""
        ),
        (
            """2 <div class ="bk bk-input-group"><input type="checkbox" """
            """name="second[0]"  class=\'bk bk-input\'/></div>"""
        ),
        (
            """3 <div class ="bk bk-input-group"><input type="checkbox" """
            """name="third[0].third" checked class='bk bk-input'/></div>"""
        )
    ]
    assert body == truth

    itms = {'first': False}
    assert opts.fromhtml(itms, txt, mdl) is None

    # pylint: disable=no-member
    assert mdl.first is False
    assert mdl.second == [False]

def test_tabkv():
    "test tab key-value parsing"

    mdl = _Dummy(
        first  = True,
        second = [False],
        third  = [_Dummy(third = [False])],
        tab    = "aaa"
    )
    txt = """
        ## tab 1 [tab:mmm]
        1 %(first)b
        ## tab 2 [tab:aaa]
        2 %(second[0])b
    """

    truth = (
        """<div class='bk bk-btn-group'><button type='button' tabkey="tab" """
        """tabvalue="mmm"  class='bk bk-btn bk-btn-default bbm-dpx-btn' """
        """id='bbm-dpx-btn-0' onclick="Bokeh.DpxModal.prototype.clicktab(0)">tab 1 """
        """</button><button type='button' tabkey="tab" tabvalue="aaa"  """
        """class='bk bk-btn bk-btn-default bbm-dpx-curbtn bk-active' """
        """id='bbm-dpx-btn-1' onclick="Bokeh.DpxModal.prototype.clicktab(1)">tab 2 """
        """</button></div><div class="bbm-dpx-hidden" id="bbm-dpx-tab-0"><table><tr >"""
        """<td>1 %(first)b</td></tr></table></div><div class="bbm-dpx-curtab" """
        """id="bbm-dpx-tab-1"><table><tr ><td>2 %(second[0])b</td></tr></table></div>"""
    )
    assert build.tohtml(txt, mdl)['body'].strip() == truth

    itms = {'tab': 'mmm'}
    assert opts.fromhtml(itms, truth, mdl) is None
    assert getattr(mdl, 'tab') == 'mmm'

    truth = (
        truth
        .replace('curbtn bk-active', '___')
        .replace('bbm-dpx-btn\'', 'bbm-dpx-curbtn bk-active\'')
        .replace('___', 'btn')
        .replace('bbm-dpx-curtab', '___')
        .replace('bbm-dpx-hidden', 'bbm-dpx-curtab')
        .replace('___', 'bbm-dpx-hidden')
    )
    assert build.tohtml(txt, mdl)['body'].strip() == truth

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

def test_build():
    "test builder"
    obj = _Dummy(aaa = "a", bbb = "b")

    def _test(txt, title, body):
        itms = build.tohtml(txt, obj, obj)

        found = ' '.join(re.split(r'\n\s*', itms['body']))
        truth = (
            ' '.join(re.split(r'\n\s*', body))
            .strip()
            .replace("> <", "><")
            .replace("> ", ">")
            .replace(" <", "<")
            .replace("  ", " ")
        )

        assert itms['title'] == title
        assert found         == truth

    body = """
        <div>
            <button type=\'button\' tabvalue='-'
                    class=\'bk bk-btn bk-btn-default bbm-dpx-curbtn bk-active\'
                    id=\'bbm-dpx-btn-0\' onclick="Bokeh.DpxModal.prototype.clicktab(0)">
                Second
            </button>
            <button type=\'button\' tabvalue='-'
                    class=\'bk bk-btn bk-btn-default bbm-dpx-btn\'
                    id=\'bbm-dpx-btn-1\' onclick="Bokeh.DpxModal.prototype.clicktab(1)">
                Third
            </button>
        </div>
        <div class="bbm-dpx-curtab" id="bbm-dpx-tab-0">
            <table></table>
        </div>
        <div class="bbm-dpx-hidden" id="bbm-dpx-tab-1">
            <table></table>
        </div>
    """
    _test(
        """
            # First

            ## Second

            ## Third
        """,
        "First",
        body
    )

    _test(
        """
            ## Second

            ## Third
        """,
        None,
        body
    )

    _test(
        """
            # Second

            # Third
        """,
        None,
        body
    )

    _test(
        """
            # Third

            aaa     %(aaa)s   %(bbb)s
            bbb               %(bbb)s

            ### AAA     YYY
            ccc         %(bbb)s
            ddd         %(bbb)s
        """,
        "Third",
        """
        <table>
            <tr >
                <td>aaa</td>
                <td >%(aaa)s</td>
                <td >%(bbb)s</td>
            </tr>
            <tr >
                <td>bbb</td>
                <td></td>
                <td >%(bbb)s</td>
            </tr>
        </table>
        <table>
            <tr style="height:20px;">
                <td></td>
                <td></td>
            </tr>
            <tr style="font-style:italic;font-weight:bold;">
                <td>AAA</td>
                <td style="font-style:italic;font-weight:normal;">YYY</td>
            </tr>
            <tr >
                <td>ccc</td>
                <td >%(bbb)s</td>
            </tr>
            <tr >
                <td>ddd</td>
                <td >%(bbb)s</td>
            </tr>
        </table>
        """
    )

def test_changelog():
    "test changelog"
    text = """
        <!DOCTYPE html>
        <html xmlns="http://www.w3.org/1999/xhtml" lang="" xml:lang="">
        <head>
          <meta charset="utf-8" />
          <meta name="generator" content="pandoc" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes" />
          <title>CHANGELOG</title>
          <style type="text/css">
              code{white-space: pre-wrap;}
              span.smallcaps{font-variant: small-caps;}
              span.underline{text-decoration: underline;}
              div.column{display: inline-block; vertical-align: top; width: 50%;}
          </style>
          <!--[if lt IE 9]>
            <script src="//cdnjs.cloudflare.com/ajax/libs/html5shiv/3.7.3/html5shiv-printshiv.min.js"></script>
          <![endif]-->
        </head>
        <body>
        <nav id="TOC">
        <ul>
        <li><a href="#cycleapp">CycleApp</a><ul>
        <li><a href="#cycles_v6.6">cycles_v6.6</a><ul>
        <li><a href="#oligos">Oligos</a></li>
        </ul></li>
        <li><a href="#cycles_v6.5">cycles_v6.5</a><ul>
        <li><a href="#peaks">Peaks</a></li>
        <li><a href="#consensus">Consensus</a></li>
        </ul></li>
        <li><a href="#rampapp">RampApp</a><ul>
        <li><a href="#ramp_v2.0">ramp_v2.0</a></li>
        <li><a href="#ramp_v1.3">ramp_v1.3</a></li>
        <li><a href="#ramp_v1.2">ramp_v1.2</a></li>
        </ul></li>
        </ul>
        </nav>
        <h1 id="cycleapp">CycleApp</h1>
        <h2 id="cycles_v6.6">cycles_v6.6</h2>
        <ul>
        <li>2019-02-25 15:41:46 +0100 (tag: cycles_v6.6.5)</li>
        <li>2019-02-08 08:41:46 +0100 (tag: cycles_v6.6.4)</li>
        <li>2019-02-06 14:30:24 +0100 (tag: cycles_v6.6.3)</li>
        <li>2019-02-06 11:11:46 +0100 (tag: cycles_v6.6.2)</li>
        <li>2019-02-06 09:34:19 +0100 (tag: cycles_v6.6.1)</li>
        <li>2019-02-05 15:54:58 +0100 (tag: cycles_v6.6)</li>
        </ul>
        <h3 id="oligos">Oligos</h3>
        <p>Bindings on the forward (backward) strand can be selected on an oligo basis by prefixing the oligo with + (-).</p>
        <h2 id="cycles_v6.5">cycles_v6.5</h2>
        <ul>
        <li>2018-12-21 11:58:19 +0100 (tag: cycles_v6.5)</li>
        </ul>
        <h3 id="peaks">Peaks</h3>
        <p>In the table, the <em>Strand</em> column now reports the binding orientation as as well the sequence around either the theoretical position or the experimental position when no theoretical position was found. The sequence is marked in bold for bindings on the positive strand and italic bold for the negative strand.</p>
        <h3 id="consensus">Consensus</h3>
        <p>The goal of this <strong>new</strong> tab is to show a consensus on all beads attached to the current hairpin. If none has been indicated, the tab is of lesser interest.</p>
        <h2 id="cycles_v6.4">cycles_v6.4</h2>
        <ul>
        <li>2018-12-13 10:58:19 +0100 (tag: cycles_v6.4.1)</li>
        <li>2018-12-12 22:38:58 +0100 (tag: cycles_v6.4)</li>
        </ul>
        <h3 id="hairpin-groups">Hairpin Groups</h3>
        <p>This new tab displays multiple beads at a time. There are 3 plots:</p>
        <ul>
        <li>A scatter plot displays beads on the x-axis and hybridisation positions on the y-axis.</li>
        <li>The two histograms display durations and rates of selected hybridization positions. The user can select positions graphically using the scatter plot. This will update the histograms.</li>
        </ul>
        <p>Beads displayed are:</p>
        <ul>
        <li>the current bead,</li>
        <li>all beads which were affected to the currently selected hairpin,</li>
        <li>unless the user discarded them from the display (2nd input box on the left).</li>
        </ul>
        <p>Computations are run in the background using 2 cores. Beads will appear automatically once computed. To disable this, go to the advanced menu and set the number of cores to zero.</p>
        <h3 id="cleaning">Cleaning</h3>
        <p>Since version 6.3, values in phase 5 which are not between median positions in phase 1 and 3 are discarded. In some situations, this leads to the bead loosing a majority of values. This can happen with the SDI when there are phase jumps in the tracking algorithm. The cleaning tab will now report such situations.</p>
        <h1 id="rampapp">RampApp</h1>
        <h2 id="ramp_v2.0">ramp_v2.0</h2>
        <ul>
        <li>2018-11-08 15:11:56 +0100 (tag: ramp_v2.0, tag: cycles_v6.0)</li>
        </ul>
        <p>Refactored completely the gui. The latter is architectured as follows:</p>
        <table style="border: 1px solid black">
        <tr>
        <td>
        <table>
        <tr>
        <td style="border-bottom: 1px solid black">
        <b>Filters</b>: bead quality
        </td>
        </tr>
        <tr>
        <td style="border-bottom: 1px solid black">
        <b>Choice</b>: the type of plots
        </td>
        </tr>
        <tr>
        <td style="border-bottom: 1px solid black">
        <b>Table</b>: status summary
        </td>
        </tr>
        <tr>
        <td style="border-bottom: 1px solid black">
        <b>Slider &amp; Table</b>: beads clustered by size
        </td>
        </tr>
        <tr>
        <td style="border-bottom: 1px solid black">
        <b>Slider</b>: providing the average amount of opened hairpins per choice of Z magnet
        </td>
        </tr>
        <tr>
        <td>
        <b>Table</b>: bead opening amount per Z magnet
        </td>
        </tr>
        </table>
        </td>
        <td style="border-left: 1px solid black">
        <p><b>Graphic</b>:</p>
        <ul>
        <li>Raw data for a single bead.</li>
        <li>Average behavior for the current bead and an average behavior of all <em>good</em> beads, with their length renormalized to 100.</li>
        <li>Average behavior for the current bead and an average behavior of all <em>good</em> beads, both without length renormalization.</li>
        </ul>
        </td>
        </tr>
        </table>
        <p>In particular the amount of opening is very different from the previous version. Instead of a number of closed beads, it’s the median amount of DNA bases which remain unaccessible because the of beads still being partially opened. This median amount is over all cycles, irrespective of the beads it belongs to. Such a computation is hoped to be more robust than the previous one, especially given the usually low number of cycles available.</p>
        <p>Average behaviors are computed for each bead by:</p>
        <ol type="1">
        <li>subtracting closing hysteresis (phases … → 3) from the opening hysteresis (phases 3 → …)</li>
        <li>considering the 25th and 75th percentiles at every available Zmag.</li>
        </ol>
        <p>The average behavior for all beads is the median across <em>good</em> beads of the 25th and 75th percentile.</p>
        <h2 id="ramp_v1.3">ramp_v1.3</h2>
        <ul>
        <li>2017-03-07 11:30:13 +0100 (tag: ramp_v1.3)</li>
        </ul>
        <p>User has now access to a slider (top-right) which allows to specify the ratio of cycles which the algorithm defines as correct to tag a bead as “good”.</p>
        <p>Note that the 2 first cycles of the track file are still automatically discarded (as in pias)</p>
        <h2 id="ramp_v1.2">ramp_v1.2</h2>
        <ul>
        <li>2017-02-22 09:00:37 +0100 (tag: ramp_v1.2)</li>
        <li>2017-02-09 16:43:47 +0100 (tag: ramp_v1.1.1)</li>
        <li><p>2017-01-24 11:11:11 +0100 (tag: ramp_v1.0.1)</p></li>
        <li>Definition of a good bead has changed: For a given bead, if less than 20% of cycles do not open and close has expected the bead is tagged good. Earlier versions of rampapp discarded a bead as soon as one of its cycle misbehaved.</li>
        <li>creates a local server: once open the application runs in your webbrowser, to open another instance of rampapp, copy the address of rampapp (usually: http://localhost:5006/call_display) into a new window</li>
        <li>generates a ramp_discard.csv file for pias</li>
        <li><p>added a third graph to display the estimated size of each hairpin in the trk files</p></li>
        </ul>
        </body>
        </html>
    """.strip().replace("        <", "<")

    out = build.changelog(StringIO(text), "CycleApp")
    # pylint: disable=line-too-long
    truth = """<div><button type='button' tabvalue='-' class='bk bk-btn bk-btn-default bbm-dpx-curbtn bk-active' id='bbm-dpx-btn-0' onclick="Bokeh.DpxModal.prototype.clicktab(0)">v6.6</button><button type='button' tabvalue='-' class='bk bk-btn bk-btn-default bbm-dpx-btn' id='bbm-dpx-btn-1' onclick="Bokeh.DpxModal.prototype.clicktab(1)">v6.5</button><button type='button' tabvalue='-' class='bk bk-btn bk-btn-default bbm-dpx-btn' id='bbm-dpx-btn-2' onclick="Bokeh.DpxModal.prototype.clicktab(2)">v6.4</button></div><div class="bbm-dpx-curtab" id="bbm-dpx-tab-0"><ul>\n<li>2019-02-25 15:41:46 +0100 (tag: cycles_v6.6.5)</li>\n<li>2019-02-08 08:41:46 +0100 (tag: cycles_v6.6.4)</li>\n<li>2019-02-06 14:30:24 +0100 (tag: cycles_v6.6.3)</li>\n<li>2019-02-06 11:11:46 +0100 (tag: cycles_v6.6.2)</li>\n<li>2019-02-06 09:34:19 +0100 (tag: cycles_v6.6.1)</li>\n<li>2019-02-05 15:54:58 +0100 (tag: cycles_v6.6)</li>\n</ul>\n<h3 id="oligos">Oligos</h3>\n<p>Bindings on the forward (backward) strand can be selected on an oligo basis by prefixing the oligo with + (-).</p>\n</div><div class="bbm-dpx-hidden" id="bbm-dpx-tab-1"><ul>\n<li>2018-12-21 11:58:19 +0100 (tag: cycles_v6.5)</li>\n</ul>\n<h3 id="peaks">Peaks</h3>\n<p>In the table, the <em>Strand</em> column now reports the binding orientation as as well the sequence around either the theoretical position or the experimental position when no theoretical position was found. The sequence is marked in bold for bindings on the positive strand and italic bold for the negative strand.</p>\n<h3 id="consensus">Consensus</h3>\n<p>The goal of this <strong>new</strong> tab is to show a consensus on all beads attached to the current hairpin. If none has been indicated, the tab is of lesser interest.</p>\n</div><div class="bbm-dpx-hidden" id="bbm-dpx-tab-2"><ul>\n<li>2018-12-13 10:58:19 +0100 (tag: cycles_v6.4.1)</li>\n<li>2018-12-12 22:38:58 +0100 (tag: cycles_v6.4)</li>\n</ul>\n<h3 id="hairpin-groups">Hairpin Groups</h3>\n<p>This new tab displays multiple beads at a time. There are 3 plots:</p>\n<ul>\n<li>A scatter plot displays beads on the x-axis and hybridisation positions on the y-axis.</li>\n<li>The two histograms display durations and rates of selected hybridization positions. The user can select positions graphically using the scatter plot. This will update the histograms.</li>\n</ul>\n<p>Beads displayed are:</p>\n<ul>\n<li>the current bead,</li>\n<li>all beads which were affected to the currently selected hairpin,</li>\n<li>unless the user discarded them from the display (2nd input box on the left).</li>\n</ul>\n<p>Computations are run in the background using 2 cores. Beads will appear automatically once computed. To disable this, go to the advanced menu and set the number of cores to zero.</p>\n<h3 id="cleaning">Cleaning</h3>\n<p>Since version 6.3, values in phase 5 which are not between median positions in phase 1 and 3 are discarded. In some situations, this leads to the bead loosing a majority of values. This can happen with the SDI when there are phase jumps in the tracking algorithm. The cleaning tab will now report such situations.</p>\n</div>
    """.strip()
    assert out == truth


if __name__ == '__main__':
    test_tabkv()
