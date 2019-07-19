#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"All sequences-related stuff"
from    pathlib     import Path
from    typing      import (
    Sequence, Union, Iterable, Tuple, Dict, ClassVar, Set, Iterator, List,
    Callable, Pattern, Match, Optional, Generator, cast
)
import  re
import  numpy       as np
from    utils       import initdefaults
from    .io         import read

PEAKS_DTYPE = [('position', 'i4'), ('orientation', 'bool')] # pylint: disable=invalid-name
PEAKS_TYPE  = Sequence[Tuple[int, bool]]                    # pylint: disable=invalid-name

def _create_comple()->np.ndarray:
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
    def peaks(cls, seq:Union[str, Path], oligs:Union[Sequence[str], str],
              flags = re.IGNORECASE) -> np.ndarray:
        """
        Returns the peak positions and orientation associated to a sequence.

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
        """
        ispath = False
        if isinstance(oligs, (dict, Path)):
            seq, oligs = oligs, seq

        if isinstance(oligs, str):
            oligs = cls.split(oligs)

        if isinstance(seq, dict):
            return ((i, cls.peaks(j, oligs)) for i, j in seq.items())

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
            return ((i, cls.peaks(j, oligs)) for i, j in read(seq))

        if len(oligs) == 0:
            return np.empty((0,), dtype = PEAKS_DTYPE)

        vals  = dict() # type: Dict[int, bool]
        vals.update(cls.__get(False, seq, oligs, flags))
        vals.update(cls.__get(True, seq, oligs, flags))
        return np.array(sorted(vals.items()), dtype = PEAKS_DTYPE)

    @classmethod
    def split(cls, oligs:str)->Sequence[str]:
        "splits a string of oligos into a list"
        return [
            OligoPathParser.END[1]    if i == OligoPathParser.END[0]   else
            OligoPathParser.START[1]  if i == OligoPathParser.START[0] else
            i
            for i in sorted(set(OligoPathParser.split(oligs)))
        ]

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
        r'(?:_|^)(?P<ol>[atgc]+)(?:\-*\dxac)*_*'
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
    def parse(
            cls,
            oligos: Union[Iterable[Union[str, Pattern, None]], str, Pattern, None],
            path:   Union[Iterable[Union[str, Path]], str, Path] = ()
    ):
        "yields oligos found in the arguments or parsed from provided paths"
        if not oligos:
            return

        tosplit: str      = ''
        stems:   Set[str] = { Path(str(i)).stem for i in cls._splat(path) }
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
                    pos      = match.span()[1] -1
                    match    = cur.search(stem, pos)
        yield from cls.split(tosplit)

    @classmethod
    def splitoligos(
            cls,
            oligos: Union[Iterable[Union[str, Pattern, None]], str, Pattern, None],
            path:   Union[Iterable[Union[str, Path]], str, Path] = ()
    ) -> List[str]:
        """
        returns a sorted list of *lowercase* oligos found in the arguments or
        parsed from provided paths
        """
        if oligos is None:
            return []
        out:   Set[str] = set(i.lower() for i in cls.parse(oligos, path))
        start: Set[str] = out & set(cls.START)
        end:   Set[str] = out & set(cls.END)

        ret: List[str] = sorted(out)
        if start:
            ret.insert(0, cls.START[0])
        if end:
            ret.append(cls.END[0])
        return ret

del _create_comple
peaks             = Translator.peaks             # pylint: disable=invalid-name
complement        = Translator.complement        # pylint: disable=invalid-name
reversecomplement = Translator.reversecomplement # pylint: disable=invalid-name
splitoligos       = OligoPathParser.splitoligos  # pylint: disable=invalid-name

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
        size  = np.zeros(len(self.sequence), dtype = 'f4') # type: ignore
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

_GC = re.compile("[gcs]", re.IGNORECASE)
def gccontent(seq):
    """Calculate G+C content, return percentage (as float between 0 and 100).

    Copes mixed case sequences, and with the ambiguous nucleotide S (G or C)
    when counting the G and C content.  The percentage is calculated against
    the full length, e.g.:

    >>> from Bio.SeqUtils import GC
    >>> GC("ACTGN")
    40.0

    Note that this will return zero for an empty sequence.
    """
    return sum(1 for _ in _GC.finditer(seq)) * 100.0 / max(1, len(seq))
