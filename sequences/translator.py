#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All sequences-related stuff

Choosing Sequences
------------------

A sequence file can be indicated using the left-most dropdown menu. The files
should in fasta format:

    > sequence 1
    aaattcgaAATTcgaaattcgaaattcg
    attcgaaaTTCGaaattcgaaattcgaa

    > sequence 2
    aaattcgaaattcgaaattcgaaattcg
    attcgaaattcgaaattcgaaattcgaa

In this case, two different sequences were provided. The line starting with `>`
should contain the name of the sequence. The next lines will all be part of the
sequence until the next line starting with `>`. The sequence can be in
uppercase or not. No checks are made on the alphabet although the software will
find peak positions only where letters 'a', 't', 'c' or 'g' or their uppercase
are used. In other words, replacing parts of the sequence by 'NNNN' ensures the
software will not use that part of the sequence without changing the latter's
size. The letter 'u' is not recognized either!

Choosing Oligos
---------------

The text box below allows setting one or more oligos. Multiple oligos should be
separated by comas. The positions found can be on *either* strands.

Complex Expressions
^^^^^^^^^^^^^^^^^^^
The following alphabet is recognized, allowing for more complex expressions:

* k: either g or t
* m: either a or c
* r: either a or g
* y: either c or t
* s: either c or g
* w: either a or t
* b: any but a
* v: any but t
* h: any but g
* d: any but c
* u: t
* n or x or .: any
* !: allows setting the blocking position to the next base, as explained bellow.
* +: can be set in front of an oligo to select bindings on the forward strand only.
* -: can be set in front of an oligo to select bindings on the backward strand only.

Structural Blockings
^^^^^^^^^^^^^^^^^^^^

Two specific positions can be added to the list:

* '0' is the baseline position. Unless the bead has a majority of non-closing
  cycles, this should be the biggest peak as the blocking occurs as many times
  as there are cycles. Such blockings are the last and lowest one in the cycle.

* 'singlestrand' (or '$' for short) is the full extent of the hairpin. Most
  beads start closing during phase 4. In some cases, they do so during phase 5.
  If they occur, such blockings are the first and highest one in the cycle.

Blocking Position in the Oligo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Positions are at the end of the oligos rather than the start as the former is
where the fork blocks. In other words, given a sequence as follows, oligo TAC
will block at position 110 rather than 108::

                  3'-CAT-5'
    5'-(...)cccatattcGTAtcgtcccat(...)-3'
            :          :
            100        110

Such a behaviour doesn't work for antibodies when, for example, looking for
'CCWGG' positions in which the 'W' is methylated. In that case one can use a
'!' to mark the position to use instead. In the following example, 'c!cwgg'
will find a position at 109 and 111::

            100         111
            :           :
    3'-(...)gggtataaaGGWCCgcagggta(...)-5'
    5'-(...)cccatatttCCWGGcgtcccat(...)-3'
            :         :
            100       109

Track dependant Oligos
^^^^^^^^^^^^^^^^^^^^^^

The oligos can be parsed from the track file names:

* 'kmer': parses the track file names to find a kmer, *i.e* a sequence of `a`,
  `t`, `c` or `g`. The accepted formats are 'xxx_atc_2nM_yyy.trk' where 'xxx_'
  and '_yyy' can be anything. The 'nM' (or 'pM') notation must come immediatly
  after the kmer. It can be upper or lower-case names indifferently.
* '3mer': same as 'kmer' but detects only 3mers
* '4mer': same as 'kmer' but detects only 4mers
* A regular expression with a group named `ol`. The latter will be used as the
  oligos.
"""
from    pathlib     import Path
from    typing      import (
    Sequence, Union, Iterable, Tuple, Dict, ClassVar, Set, Iterator, List,
    Callable, Pattern, Match, Optional, Generator, cast
)
import  re
import  numpy       as np
from    utils       import initdefaults
from    .io         import read

PEAKS_DTYPE = [('position', 'i4'), ('orientation', 'bool')]  # pylint: disable=invalid-name
PEAKS_TYPE  = Sequence[Tuple[int, bool]]                     # pylint: disable=invalid-name
Oligos      = Union[Iterable[Union[str, Pattern, None]], str, Pattern, None]
Sequences   = Union[Dict[str, str], str, Path]
Paths       = Union[Iterable[Union[str, Path]], str, Path]

def _create_comple() -> np.ndarray:
    out = np.zeros(256, dtype = '<U1')
    for _ in range(256):
        out[_] = chr(_)

    for i, j in ('at', 'cg', 'km', 'ry', 'bv', 'hd'):
        out[ord(i)] = j
        out[ord(j)] = i
        out[ord(i.upper())] = j.upper()
        out[ord(j.upper())] = i.upper()

    out[ord('u')] = 'a'
    out[ord('U')] = 'A'
    return out

class Translator:
    "Translates a sequence to peaks"
    __SYMBOL: ClassVar[str] = '!'
    __STATE:  ClassVar[str] = '-+'
    __METHS:  ClassVar[Tuple[Tuple[Pattern, Callable[[Match],str]],...]]  = (
        (re.compile('.'+__SYMBOL), lambda x: '('+x.string[x.start()]+')'),
        (re.compile(__SYMBOL+'.'), lambda x: '('+x.string[x.end()-1]+')')
    )
    __TRANS: ClassVar[Dict[str, str]] = {
        'k': '[gt]', 'm': '[ac]', 'r': '[ag]', 'y': '[ct]', 's': '[cg]',
        'w': '[at]', 'b': '[^a]', 'v': '[^t]', 'h': '[^g]', 'd': '[^c]',
        'n': '.',    'x': '.', 'u': 't'
    }
    __TRANS.update({i.upper(): j for i, j in __TRANS.items()})

    __TRAFIND: ClassVar[Pattern]   = re.compile('['+''.join(__TRANS)+']')
    __COMPLE:  ClassVar[np.ndarray] = _create_comple()

    @classmethod
    def __trarep(cls, item):
        return cls.__TRANS[item.string[slice(*item.span())]]

    @classmethod
    def __translate(cls, olig, state):
        if not state:
            olig = cls.reversecomplement(olig)
        if cls.__SYMBOL in olig:
            olig = cls.__METHS[state][0].sub(cls.__METHS[state][1], olig)
        return cls.__TRAFIND.sub(cls.__trarep, olig)

    @classmethod
    def reversecomplement(cls, oligo: str) -> str:
        "returns the reverse complement for that oligo"
        return oligo[::-1].translate(cls.__COMPLE)

    @classmethod
    def complement(cls, oligo: str) -> str:
        "returns the complement for that oligo"
        return oligo.translate(cls.__COMPLE)

    @classmethod
    def __get(cls, state, seq, oligs, flags):
        for oli in oligs:
            if len(oli) == 0:
                continue

            if oli[0] in cls.__STATE:
                if state ^ (oli[0] == cls.__STATE[1]):
                    continue
            oli = oli.replace(cls.__STATE[0], '').replace(cls.__STATE[1], '')

            if oli == OligoPathParser.START[1]:
                if state:
                    yield (0, state)
                continue

            if oli == OligoPathParser.END[1]:
                if state:
                    yield (len(seq), state)
                continue

            patt = cls.__translate(oli, state)
            reg  = re.compile(patt, flags)
            val  = reg.search(seq, 0)

            cnt  = range(1, patt.count('(')+1)
            if '(' in patt:
                while val is not None:
                    spans = [val.span(i)[-1] for i in cnt]
                    yield from ((i, state) for i in spans if i > 0)
                    val = reg.search(seq, val.start()+1)
            else:
                while val is not None:
                    yield (val.end(), state)
                    val = reg.search(seq, val.start()+1)

    @classmethod
    def peaks(
            cls,
            seq:    Sequences,
            oligos: Oligos,
            path:   Paths = (),
            flags = re.IGNORECASE
    ) -> np.ndarray:
        """
        Returns the peak positions and orientation associated to one or more
        sequence, directly as a np.ndarray in the first case and as an iterator
        in the second.

        A peak position is the end position of a match. With indexes starting at 0,
        that's the indexe of the first base *after* the match.

        The orientation is *True* if the oligo was matched and false otherwise. Palindromic
        cases are *True*.

        Matches are **case sensitive**.

        Example:

        ```python
        import numpy as np
        seq = "atcgATATATgtcgCCCaaGGG"

        res = peaks(seq, ('ATAT', 'CCC'))
        assert len(res) == 4
        assert all(a == b for a, b in zip(res['position'],    [8, 10, 17, 22]))
        assert all(a == b for a, b in zip(res['orientation'], [True]*3+[False]))

        res = peaks(seq, 'ATAT')
        assert len(res) == 2
        assert all(a == b for a, b in zip(res['position'],    [8, 10]))
        assert all(a == b for a, b in zip(res['orientation'], [True]*2))

        seq = "c"*5+"ATC"+"g"*5+"TAG"+"c"*5
        res = peaks(seq, 'wws')
        assert len(res) == 4
        ```

        Parameters
        ----------
        sequences:
            The sequence or sequences through which to look for the oligos.
            These may be a single sequence, a dictionnary or a path to a fasta
            file.
        oligos:
            The oligos, either as a keyword ('kmer', '3mer', ...), a string of
            comma-separated oligos, a pattern to use for parsing the track file
            names.
        path:
            The paths to parse for oligos.
        """
        ispath = False
        if isinstance(oligos, (dict, Path)):
            seq, oligos = oligos, seq

        olist = OligoPathParser.splitoligos(oligos, path = path)
        if isinstance(seq, dict):
            return ((i, cls.peaks(j, olist)) for i, j in seq.items())

        ispath = isinstance(seq, Path)
        if (
                isinstance(seq, str)
                and any(i in seq for i in '/.\\')
                and len(seq) <= 1024
        ):
            try:
                ispath = Path(seq).exists()
            except OSError:
                pass

        if ispath:
            return ((i, cls.peaks(j, olist)) for i, j in read(seq))

        if len(olist) == 0:
            return np.empty((0,), dtype = PEAKS_DTYPE)

        vals  = dict()  # type: Dict[int, bool]
        vals.update(cls.__get(False, seq, olist, flags))
        vals.update(cls.__get(True, seq, olist, flags))
        return np.array(sorted(vals.items()), dtype = PEAKS_DTYPE)

    @classmethod
    def split(cls, *args, **kwa) -> List[str]:
        """"
        Splits a string of oligos into a list and returns the sorted oligos.
        """
        return OligoPathParser.splitoligos(*args, **kwa)

class OligoPathParser:
    "Translates a sequence to peaks"
    START:  ClassVar[Tuple[str,...]] = (
        '0', 'start', 'first', 'zero', 'doublestrand', 'closed'
    )
    END:    ClassVar[Tuple[str,...]] = (
        '_', 'singlestrand', '$', '-1', 'last', 'end', 'open'
    )
    _ABC:   ClassVar[str]            = r'\w_\$!\+\-'
    _SPLIT: ClassVar[Pattern]       = re.compile(
        rf'(?:[^{_ABC}]*)([{_ABC}]+)(?:[^{_ABC}]+|$)*', re.IGNORECASE
    )

    _MER:  ClassVar[str]      = (
        r'(?:_|^)(?P<ol>[atgc]+)(?:\-*\dxac)*_*(?:2amino(?:_*datp)*|[dlbr]na)*_*'
        r'(?P<id>\d+(?:[.dp]\d+)*)(?P<unit>[np]M)(?:_|$)'
    )
    _KMER: ClassVar[Pattern] = re.compile(_MER, re.IGNORECASE)
    _3MER: ClassVar[Pattern] = re.compile(
        _MER.replace('[atgc]+', '[atgc]'*3), re.IGNORECASE
    )
    _4MER: ClassVar[Pattern] = re.compile(
        _MER.replace('[atgc]+', '[atgc]'*4), re.IGNORECASE
    )
    _5MER: ClassVar[Pattern] = re.compile(
        _MER.replace('[atgc]+', '[atgc]'*5), re.IGNORECASE
    )
    _KMERS: ClassVar[Pattern] = _KMER
    _3MERS: ClassVar[Pattern] = _3MER
    _4MERS: ClassVar[Pattern] = _4MER
    _5MERS: ClassVar[Pattern] = _5MER

    @classmethod
    def split(cls, oligs:str) -> Iterator[str]:
        "splits a string of oligos into a list"
        for i in cls.START[1:]:
            oligs = oligs.replace(i, cls.START[0]).replace(i.upper(), cls.START[0])
        for i in cls.END[1:]:
            oligs = oligs.replace(i, cls.END[0]).replace(i.upper(), cls.END[0])
        yield from cls._SPLIT.findall(oligs)

    _ITEMS = (tuple, list, set, frozenset, Iterator, Generator)
    @classmethod
    def _splat(cls, items) -> Iterator:
        rem = list(items) if isinstance(items, cls._ITEMS) else [items]
        while rem:
            cur = rem.pop()
            if not cur:
                continue
            if isinstance(cur, cls._ITEMS):
                rem.extend(cur)
            else:
                yield cur

    @classmethod
    def parse(cls, oligos: Oligos, path: Paths = ()):
        "yields oligos found in the arguments or parsed from provided paths"
        if not oligos:
            return

        tosplit: str      = ''
        stems:   Set[str] = {Path(str(i)).stem for i in cls._splat(path)}
        cur:     Union[str, Pattern]
        for cur in set(cls._splat(oligos)):
            if isinstance(cur, str):
                cur = (
                    re.compile(cur) if '(?P<' in cur else
                    getattr(cls, '_'+cur.strip().upper(), cur)
                )
                if isinstance(cur, str):
                    tosplit += ','+cur
                    continue

            for stem in stems:
                pos:   int             = 0
                match: Optional[Match] = cur.search(stem)
                while match is not None:
                    tosplit += ','+match.group('ol')
                    pos      = match.span()[1] - 1
                    match    = cur.search(stem, pos)
        yield from cls.split(tosplit)

    @classmethod
    def splitoligos(
            cls,
            oligos:       Oligos,
            path:         Paths = (),
    ) -> List[str]:
        """
        Returns a sorted list of *lowercase* oligos found in the arguments or
        parsed from provided paths

        Parameters
        ----------
        oligos:
            The oligos, either as a keyword ('kmer', '3mer', ...), a string of
            comma-separated oligos, a pattern to use for parsing the track file
            names.
        path:
            The paths to parse for oligos.
        """
        if oligos is None:
            return []
        out:   Set[str]  = set(i.lower() for i in cls.parse(oligos, path))
        start: Set[str]  = out & set(cls.START)
        end:   Set[str]  = out & set(cls.END)

        ret:   List[str] = sorted(out-start-end)
        if start:
            ret.insert(0, cls.START[1])
        if end:
            ret.append(cls.END[1])
        return ret


del _create_comple
peaks             = Translator.peaks              # pylint: disable=invalid-name
complement        = Translator.complement         # pylint: disable=invalid-name
reversecomplement = Translator.reversecomplement  # pylint: disable=invalid-name
splitoligos       = OligoPathParser.splitoligos   # pylint: disable=invalid-name


def marksequence(seq:str, oligs: Sequence[str]) -> str:
    u"Returns a sequence with oligos to upper case"
    seq = seq.lower()
    for olig in oligs:
        seq  = seq.replace(olig.lower(), olig.upper())

        olig = Translator.reversecomplement(olig)
        seq  = seq.replace(olig.lower(), olig.upper())
    return seq

def markedoligos(oligos: Sequence[str]) -> Dict[str, str]:
    "Return a dictionnary of oligos and the characters with which to replace them"
    oligs = {
        i.replace('!', '').lower()
        for i in oligos if i.lower() not in ('$', '0', 'doublestrand', 'singlestrand')
    }

    lst = {i: ''.join(chr(0x1d58d+ord(j)) for j in i) for i in oligs}

    oligs = {reversecomplement(i) for i in oligs}-oligs
    lst.update({i: ''.join(chr(0x1d5f5+ord(j)) for j in i) for i in oligs})
    return lst

def overlap(ol1:str, ol2:str, minoverlap = None):
    """
    Returns wether the 2 oligos overlap

    Example:

        >>> import numpy as np
        >>> assert  not overlap('ATAT', '')
        >>> assert  overlap('ATAT', 'ATAT')
        >>> assert  overlap('ATAT', 'CATA')
        >>> assert  overlap('ATAT', 'CCAT')
        >>> assert  overlap('ATAT', 'CCCA')
        >>> assert  overlap('ATAT', 'ATAT', minoverlap = 4)
        >>> assert  overlap('ATAT', 'CATA', minoverlap = 3)
        >>> assert  overlap('ATAT', 'CCAT', minoverlap = 2)
        >>> assert  overlap('ATAT', 'CCCA', minoverlap = 1)
        >>> assert  not overlap('ATAT', 'ATAT', minoverlap = 5)
        >>> assert  not overlap('ATAT', 'CATA', minoverlap = 4)
        >>> assert  not overlap('ATAT', 'CCAT', minoverlap = 3)
        >>> assert  not overlap('ATAT', 'CCCA', minoverlap = 2)

        >>> assert  not overlap('', 'ATAT')
        >>> assert  overlap('ATAT', 'ATAT')
        >>> assert  overlap('CATA', 'ATAT')
        >>> assert  overlap('CCAT', 'ATAT')
        >>> assert  overlap('CCCA', 'ATAT')
        >>> assert  overlap('ATAT', 'ATAT', minoverlap = 4)
        >>> assert  overlap('CATA', 'ATAT', minoverlap = 3)
        >>> assert  overlap('CCAT', 'ATAT', minoverlap = 2)
        >>> assert  overlap('CCCA', 'ATAT', minoverlap = 1)
        >>> assert  not overlap('ATAT', 'ATAT', minoverlap = 5)
        >>> assert  not overlap('CATA', 'ATAT', minoverlap = 4)
        >>> assert  not overlap('CCAT', 'ATAT', minoverlap = 3)
        >>> assert  not overlap('CCCA', 'ATAT', minoverlap = 2)

    """
    if len(ol1) < len(ol2):
        ol1, ol2 = ol2, ol1

    if minoverlap is None or minoverlap <= 0:
        minoverlap = 1

    if minoverlap > len(ol2):
        return False

    rng = range(minoverlap, len(ol2))
    if ol2 in ol1:
        return True
    return any(ol1.endswith(ol2[:i]) or ol1.startswith(ol2[-i:]) for i in rng)

class NonLinearities(Translator):
    """
    Computes the non-linearities created by oligos binding to a hairpin
    """
    sequence     = None
    singlestrand = .55  # nm
    doublestrand = .34  # nm
    orientation  = None
    @initdefaults(frozenset(locals()))
    def __init__(self, **_):
        super().__init__()

    def difference(self, oligos: Union[str, Iterable[str]]):
        """
        returns the difference to the a single strand in nm
        """
        size  = np.zeros(len(self.sequence), dtype = 'f4')  # type: ignore
        diff  = self.doublestrand - self.singlestrand
        allo  = (oligos,) if isinstance(oligos, str) else oligos
        for olig in cast(Iterable[str], allo):
            pks  = self.peaks(cast(str, self.sequence), olig)
            good = pks['position']
            if self.orientation is not None:
                good = good[pks['orientation'] == self.orientation]
            for i in good:
                size[i:] += diff
        return size

    def nonlinearities(self, oligos: Union[str, Iterable[str]]):
        """
        returns the non-linearities created by the oligos in nm
        """
        size   = self.difference(oligos)
        single = np.arange(len(size), dtype = 'f4') * self.singlestrand
        double = size+single
        return double - np.polyval(np.polyfit(single, double, 1), single)


def gccontent(seq, _gc_ = re.compile("[gcs]", re.IGNORECASE)):
    """Calculate G+C content, return percentage (as float between 0 and 100).

    Copes mixed case sequences, and with the ambiguous nucleotide S (G or C)
    when counting the G and C content.  The percentage is calculated against
    the full length, e.g.:

    >>> from Bio.SeqUtils import GC
    >>> GC("ACTGN")
    40.0

    Note that this will return zero for an empty sequence.
    """
    return sum(1 for _ in _gc_.finditer(seq)) * 100.0 / max(1, len(seq))
