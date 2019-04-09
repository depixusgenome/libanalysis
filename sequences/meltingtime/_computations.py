#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
Computing melting temperature, Kon and Koff

The computations rely on a Markov model. Each state is a combination of bound
*double* pairs of bases. This means nucleotides are always *bound* by at least
NpN bases, meaning 4 bases in all. These 4 bases may include one miss-match.
Two missmatches is a program error. For each 4 bases, an enthalpy and an
enthropy is computed depending on their proximity to the edges and their
composition.

The fork effect is simulated by simply considering that the 5' end of the
hairpin is always hybridized. There are thus 1/2 as many states available for
the hairpin as there are for other pairs of oligos (which can dehybridize from
both ends).

Transitions are computed considering the enthalpy/enthropy delta betwen the
current state and a state where one more base is either hybridized or
dehybridized. Thus states accessible directly from a given state are only those
with a single difference in the hybridized base pairs.

Enthalpies and enthropies are provided in the module *._data*. Values are
available for:

* DNA-DNA pairs of bases withouth missmatch. Keys are 'WX/YZ' with WX on the 5'
end. The letters must be upper-case.
* DNA-DNA pairs of bases with internal missmatch. Keys are 'WX/YZ' with WX on the 5'
end. The letters must be upper-case.
* DNA-DNA pairs of bases with terminal missmatch. Keys are 'WX./YZ.' with WX on the 5'
end. The letters must be upper-case.
* The same for LNA-DNA and LNA-LNA, the LNA being marked with a lower case letter.
* The same for RNA-DNA and RNA-RNA, format unknown.

The information for oligo and hairpint sequences should be provided in
upper-case for DNA bases and lower-case for LNA.

An example:
```python
    cnf = TransitionStats(
        "CCCCTAGGGGATTACCC",  # hairpin 5'->3' sequence
        ("GATC",  False, 3),  # bind an oligo to the 5' strand at 3 bases from the start
        ('TAAT',  False, 10), # bind an oligo to the 5' strand at 10 bases from the start
        ('AGGGA', True,  5),  # bind an oligo to the 3' strand at 5 bases from the start
        force = 8.5           # force exerted on the hairpin
    )

    cnf.statistics()   #  -->  trep, koff, kon, dgf melting T°

    assert str(cnf.strands) == \"\"\"
        hairpin: CCCCTAGGGGATTACCC
        -     0: ...GATC..........
        -     2: ..........TAAT...
        +     1: .....AGGGA.......
                 GGGGATCCCCTAATGGG
    \"\"\"
```
"""
from   enum          import Enum
from   itertools     import chain, product
from   typing        import Iterator, Tuple, Optional, List, Callable, cast
import re
import numpy  as np
from   numpy.linalg  import solve as _solve
from   ._base        import BaseComputations, complement, gccontent, SaltInfo
def _frombuffer(*buffers):
    return (
        np.frombuffer(
            buf,
            dtype = [(f'f{i}', 'i4') for i in range(buf.shape[1])]
        )
        for buf in buffers
    )

class InitialState(Enum):
    "Compuptation mode of the initial state"
    stable     = "stable"
    hybridized = "hybridized"
    @classmethod
    def _missing_(cls, value):
        return InitialState('stable') if value is None else super()._missing_(value)

class Strands:
    "all about strands"
    def __init__(
            self,
            seq:        Optional[str],
            opposite:   Optional[str] = None,
            shift:      int           = 0,
            density:    float         = 25
    ):
        if isinstance(seq, (list, tuple)) and opposite is None:
            seq, opposite = seq

        if opposite is None:
            opposite = complement(cast(str, seq))
        elif seq is None:
            seq      = complement(cast(str, opposite))
        assert isinstance(seq,      str)
        assert isinstance(opposite, str)

        seq      = '.' * max(0, -shift) + seq
        opposite = '.' * max(0, shift)  + opposite
        dsfcn    = lambda x, y: next(i for i, j in enumerate(zip(x, y)) if '.' not in j)

        self.seq      : str   = seq    + '.' * max(0, len(opposite) - len(seq))
        self.opposite : str   = opposite  + '.' * max(0, len(seq)   - len(opposite))
        self.shift    : int   = shift
        self.density  : float = density
        self.dsbegin  : int   = dsfcn(self.seq, self.opposite)
        self.dsend    : int   = self.maxsize-dsfcn(self.seq[::-1], self.opposite[::-1])

    @property
    def maxsize(self) -> int:
        "return the strand max size"
        return len(self.seq)

    @property
    def dssize(self) -> int:
        "return the double strand size"
        return len(self.dsseq)

    @property
    def dsseq(self):
        "return the double stranded part of the sequence"
        return self.seq[self.dsbegin:self.dsend]

    @property
    def dsopposite(self):
        "return the double stranded part of the opposite sequence"
        return self.opposite[self.dsbegin:self.dsend]

    @property
    def doublestrand(self):
        "return the double stranded part"
        return self.dsseq, self.dsopposite

    def key(self, ind:int) -> str:
        "get table keys"
        return self.opposite[ind:ind+2] + '/' + self.seq[ind:ind+2]

    def terminalbases(self) -> Iterator[Tuple[int, str]]:
        "return keys for terminal bases"
        start, end = self.dsbegin, self.dsend-2
        seq,   opp = self.seq,     self.opposite
        yield (start, f'{seq[start:start+2][::-1]}./{opp[start:start+2][::-1]}.')
        if end > start:
            yield (end, f'{opp[end:end+2]}./{seq[end:end+2]}.')

    def danglingends(self) -> Iterator[Tuple[int, str]]:
        "return keys for terminal bases"
        for side in (False, True):
            ind = self.dsend-1 if side else max(self.dsbegin-1, 0)
            if any(i[ind] == "." for i in (self.seq, self.opposite)):
                if side:
                    yield (
                        ind-1,
                        self.seq[ind-1:ind+1][::-1] + '/' + self.opposite[ind-1:ind+1][::-1]
                    )
                else:
                    yield (
                        ind,
                        self.opposite[ind:ind+2] + '/' + self.seq[ind:ind+2]
                    )

    def dsbases(self, keys) -> Iterator[Tuple[int, str]]:
        """
        return keys for the double stranded part of the dna

        Args:
          * keys: the available terminal missmatch keys. When unavailable,
          we'll automatically fall back on internal missmatches.
        """
        start = self.dsbegin
        bases = [i[1] for i in self.terminalbases()]
        if bases[0] in keys:
            start += 1
        stop  = self.dsend-1
        if bases[1] in keys:
            stop -= 1
        yield from ((i, self.key(i)) for i in range(start, stop))

    @property
    def isoligo(self):
        "return whether the opposite sequence is shortker than the other"
        return self.seq[0] == '.' and self.seq[-1] == '.'

    @property
    def iscoligo(self):
        "return whether the opposite sequence is shorter than the other"
        return self.opposite[0] == '.' and self.opposite[-1] == '.'

    @property
    def reversecomplement(self) -> 'Strands':
        "return the reverse complement of the strands"
        cpy = type(self)("", "")
        cpy.seq      = complement(self.opposite)[::-1]
        cpy.opposite = complement(self.seq)[::-1]
        cpy.shift    = (self.dsend - self.maxsize) * (1 if self.shift > 0 else -1)
        cpy.dsbegin  = self.maxsize-self.dsend
        cpy.dsend    = self.maxsize-self.dsbegin
        return cpy

    def nstates(self, nzipped = -1) -> int:
        """
        The number of different states provided *both* ends can de-hybredize.

        The argument *nzipped* tells how many bases are zipped starting from the
        left: this is the effect of the fork.
        """
        assert self.dsbegin > 1 and self.dsend < self.maxsize
        nds = max(0, self.dsend - max(nzipped+1, self.dsbegin))
        return (nds*(nds-1))//2+1

_OFF:        int      = cast(int, np.iinfo('i4').max)
_DTYPE:      np.dtype = np.dtype('f8')
TRANSITIONS: np.dtype = np.dtype([
    ('states',   np.dtype([('iclose',  'i4'), ('iopen',    'i4')])),
    ('base',     'i4'),
    ('islast',   'i1'),
    ('canclose', 'bool'),
    ('canopen',  'bool'),
    ('_',        'i1')
])

class StrandList:
    "Compute the state list for complex states"
    _MATCH = re.compile(r"([53])-(\d+)(.*)").match
    def __init__(self, *seqs, **_):
        """
        SEQS arguments work as follows:

        * starting with a '5-' is an oligo binding to the 5' strand
        * starting with a '3-' is an oligo binding to the 3' strand
        * following the '5-' or '3-' must come a number indicating the position
          relative to the hairpin
        * if only '5-' '3-' are indicated or if both these and numbers are
        missing, this is expected to be the hairpin's 5' (then 3') sequence
        """
        hpin:   Optional[str] = None
        chpin:  Optional[str] = None
        oligos: List          = []
        for arg in seqs:
            if isinstance(arg, (tuple, list)) and len(arg) == 2 and None in arg:
                hpin, chpin = arg
                continue
            if isinstance(arg, (tuple, list, dict)):
                oligos.append(arg)
                continue

            assert isinstance(arg, str)
            elem = self._MATCH(arg)
            if elem:
                oligos.append((elem.group(3), elem.group(1) == '5', int(elem.group(2))))
            elif arg[:2] == '5-':
                hpin = arg[2:]
            elif arg[:2] == '3-':
                chpin = arg[2:]
            elif hpin is not None:
                chpin = arg
            else:
                hpin  = arg

        self.hpin = Strands(hpin, opposite = chpin)
        def _new(oligo:str, seq:bool = False, shift:int = 0) -> Strands:
            "create a new strand"
            return (
                Strands(seq = oligo, opposite = self.hpin.opposite, shift = -abs(shift))
                if seq else
                Strands(seq = self.hpin.seq, opposite = oligo, shift = abs(shift))
            )

        self.oligos = sorted(
            [
                _new(*i)  if isinstance(i, (list, tuple)) else
                _new(**i) if isinstance(i, dict) else
                _new(i)
                for i in oligos
            ],
            key = lambda x: x.dsbegin
        )

        test = lambda ols: np.all(
            np.array([i.dsend for i in ols[:-1]]) < np.array([i.dsbegin for i in ols[1:]])
        )
        assert test([i for i in self.oligos if i.isoligo])
        assert test([i for i in self.oligos if i.iscoligo])

    def nstates(self, *masked) -> np.ndarray:
        """
        number of different states
        """
        return sum(i for _, i in self.__index_info(masked)[-1])

    def states(self, *masked) -> np.ndarray:
        """
        all the different states
        """
        hpin, inner, ind, itr = self.__index_info(masked)
        outs      = np.zeros((self.nstates(*masked), 1 + hpin + len(inner)), dtype = "i4")
        outs[:,0] = np.arange(outs.shape[0], dtype = 'i4')
        rem       = outs[:,1:]
        for i, cnt in itr:
            if hpin:
                rem[:cnt, 0] = i
            rem[:cnt, hpin:] = list(product(*(k[ind(i, j, l):] for j, k, l in inner)))
            rem              = rem[cnt:,:]

        return outs

    @classmethod
    def hairpintransitions(cls, states: np.ndarray) -> np.ndarray:
        "return the hairpin transitions: an array of TRANSITIONS"
        ison                          = states[:,1] > 0
        aopened                       = states[:,1:][ison]
        aopened[:,0]                 -= 1
        aopened[aopened[:,0] == 1, 0] = 0 # require 2 bases to hold the hp

        out            = cls.__join(states, aopened, ison, 1)
        out['canopen'] = (out['base'] > 2) & (out['base'] < states[:,1].max())
        out['base']    = np.maximum(out['base']-2, 0)
        return out

    @classmethod
    def oligotransitions(cls, maxs:int, states, isright: bool, key:int) -> np.ndarray:
        "return the oligo transitions: an array of TRANSITIONS"
        return (cls.__join_oligo_right if isright else cls.__join_oligo_left)(maxs, states, key)

    def transiantstates(self, *masked, states: Optional[np.ndarray] = None) -> np.ndarray:
        "states to be used for computations"
        states = self.parseargs(masked, states)[1]
        if self.hpin in masked:
            return states[:,1:].max(1) != _OFF
        return states[:,1] != states[:,1].max()

    def representation(self):
        "return a string representation of the oligos"
        seqs = f"hairpin: {self.hpin.seq}"
        opps = ""
        for i, j in enumerate(self.oligos):
            if '.' in j.seq:
                opps +=  f"\n-{i: 6d}: {j.seq}"
            else:
                seqs +=  f"\n+{i: 6d}: {j.opposite}"
        return f"{seqs}{opps}\n         {self.hpin.opposite}"

    def __str__(self):
        return f"{super().__str__()}\n{self.representation()}"

    def __index_info(
            self,
            masked:Tuple[Strands,...]
    ) -> Tuple[
        bool,
        List[Tuple[np.ndarray, np.ndarray, int]],
        Callable,
        Iterator[Tuple[int, int]]
    ]:
        hpin = self.hpin not in masked
        ols  = [i for i in self.oligos if i not in masked] if masked else self.oligos
        maxs = self.ndigits
        assert maxs < _OFF and (maxs**2) < _OFF # type: ignore

        inner = [
            (
                np.cumsum([i.dsend-j-1 for j in range(i.dsbegin, i.dsend-1)], dtype = 'i4'),
                np.append(
                    np.fromiter(
                        (
                            k
                            for j in range(i.dsbegin*maxs, (i.dsend-1)*maxs,maxs)
                            for k in range(j+j//maxs+2, j+i.dsend+1)
                        ),
                        dtype = 'i4',
                        count = (i.dsend-i.dsbegin-1)*(i.dsend-i.dsbegin) // 2
                    ),
                    _OFF
                ),
                i.dsbegin
            ) for i in ols
        ]

        ind = lambda i, j, l: j[min(len(j)-1, i-l-1)] if i > l else 0
        itr = (
            (i, np.prod([len(k)-ind(i, j, l) for j, k, l in inner]))
            for i in (chain(range(1), range(2,len(self.hpin.seq)+1)) if hpin else range(-2,-1))
        )
        return hpin, inner, ind, itr

    @property
    def ndigits(self) -> int:
        "number of digits per index indicating the start/end of a hybridization"
        return 10**(int(np.round(np.log10(self.hpin.maxsize)))+1)

    def parseargs(
            self,
            masked: tuple,
            states: Optional[np.ndarray]
    ) -> Tuple[Tuple[Strands,...], np.ndarray]:
        "returns the masked strands and the list of states"
        if states is None:
            states = next((i for i in masked if isinstance(i, np.ndarray)), None)
            masked = tuple(i for i in masked if not isinstance(i, np.ndarray))

        masked = tuple(
            i              if isinstance(i, Strands)                          else
            self.oligos[i] if isinstance(i, int)                              else
            self.hpin      if i in ('hpin', 'seq', 'template', self.hpin.seq) else
            self.oligos[0] if i == 'oligo'                                    else
            next(j for j in self.oligos if j.seq == i)
            for i in masked
        )

        if states is None:
            states = self.states(*masked)
        return masked, states

    @classmethod
    def __join(
            cls,
            states:  np.ndarray,
            aopened: np.ndarray,
            ison:    np.ndarray,
            key:     int
    ) -> np.ndarray:
        closed, opened = _frombuffer(np.copy(states[:,1:]), aopened)
        inds           = np.searchsorted(closed, opened)
        good           = closed[inds] == opened
        inds           = inds[good]

        keys = [0, 0, key, key]
        out  = np.frombuffer(states[ison, :][:, keys][good], dtype = TRANSITIONS)
        out['states']['iopen']       = states[inds, :][:,0]
        out[['canclose', 'canopen']] = True
        out['islast']                = False
        return out

    @staticmethod
    def __ndoublestrands(maxs:int, out: np.ndarray):
        return (out % maxs) - (out // maxs)

    @classmethod
    def __islast(cls, maxs:int, out: np.ndarray):
        out['islast'] = cls.__ndoublestrands(maxs, out['base']) <= 3 | (out['base'] == _OFF)

    @classmethod
    def __join_oligo_left(cls, maxs:int, states, key:int) -> np.ndarray:
        ison    = states[:,key] != _OFF
        aopened = states[:,1:][ison]
        aopened[:,key-1]                                                 += maxs
        aopened[cls.__ndoublestrands(maxs, aopened[:,key-1]) <= 1, key-1] = _OFF

        out = cls.__join(states, aopened, ison, key)
        cls.__islast(maxs, out)
        out['canclose'] = states[out['states']['iopen'], key] != _OFF
        out['base']   //= maxs
        return out

    @classmethod
    def __join_oligo_right(cls, maxs:int, states, key:int) -> np.ndarray:
        # Dehibridiations are counted twice,
        # once from the left, once from the right.
        # We arbitrarily discard the right occurence
        ison    = (cls.__ndoublestrands(maxs, states[:, key]) > 2)  & (states[:,key] != _OFF)
        aopened = states[:,1:][ison]
        aopened[:,key-1] += -1

        out = cls.__join(states, aopened, ison, key)
        cls.__islast(maxs, out)
        out['base'] = np.maximum((out['base'] % maxs)-2, states[:,key].min()//maxs)
        return out

class EnergyComputations(BaseComputations):
    "computes the enthalpies & entropies per base"
    def saltinfo(self, hpin: Strands, strands: Strands, dtype: np.dtype = _DTYPE) -> SaltInfo:
        "return the salt info for a given oligo"
        delta = self.__delta(strands, dtype)
        return self.salt.compute(
            strands.dsseq,
            hpin.density,
            strands.density,
            self.temperatures.mtG,
            delta
        )

    def roo(self, hpin: Strands, strands:Strands, dtype: np.dtype = _DTYPE):
        "create the roo vector"
        sli        = slice(strands.dsbegin, strands.dsend-1)

        roo        = np.zeros((strands.maxsize,2), dtype = dtype)
        roo[sli,0] = -self.__dgperbase(hpin, strands, dtype)[sli]

        dgt0       = self.dgcor(self.table['init_A/T'] - self.table['init_G/C'])
        if strands.seq[cast(int, sli.start)] in 'AaTt':
            roo[sli.start,0]             += -dgt0
        if strands.seq[cast(int, sli.stop)] in 'AaTt':
            roo[cast(int, sli.stop)-1,0] += dgt0

        roo[sli,1] += roo[sli,0]-self.dgcor(self.table['init_G/C'])
        roo[sli,:]  = np.exp(roo[sli,:])
        return np.kron(roo, [-1., 1.]).reshape((-1, 2, 2))

    def rch(self, dtype: np.dtype = _DTYPE) -> np.ndarray:
        "return the hairpin closing r"
        return self.elasticity.rch(self.temperatures)*np.array([1, -1], dtype = dtype)

    def rco(self, dtype: np.dtype = _DTYPE) -> np.ndarray:
        "return the oligos' closing r"
        return self.elasticity.rco(self.temperatures)*np.array([1, -1], dtype = dtype)

    def temperaturesanddgf(
            self,
            hpin:    Strands,
            strands: Strands,
            dtype:   np.dtype = _DTYPE
    ) -> Tuple[float, float]:
        "return the melting temperature and the dgf"
        temp, delta = self.saltinfo(hpin, strands, dtype)[2:]
        return temp, self.cor*delta+(strands.dssize-1)*(self.gss-self.gds)

    def __delta(self, strands:Strands, dtype: np.dtype = _DTYPE) -> np.ndarray:
        "compute the deltas"
        dsseq   = strands.dsseq
        delta    = np.array([0., 0.], dtype = dtype)

        # Type: General initiation value
        delta += self.table['init']

        # Type: Duplex with no (allA/T) or at least one (oneG/C) GC pair
        delta += self.table['init_'+('oneG/C' if gccontent(dsseq.upper()) else 'allA/T')]

        # Type: Penalty if 5' end is T
        delta += self.table['init_5T/A'] * ((dsseq[0] in 'Tt')+(dsseq[-1] in 'Aa'))

        # Type: Different values for G/C or A/T terminal basepairs
        ends = (dsseq[0] + dsseq[-1]).upper()
        for tpe in ("AT", "GC"):
            tmp   = self.table[f'init_{tpe[0]}/{tpe[1]}']
            delta += sum(ends.count(i) for i in tpe) * tmp

        # Terminal endings
        for _, key in chain(strands.terminalbases(), strands.danglingends()):
            tmp = self.table.get(key, None)
            if tmp is not None:
                delta += tmp

        # inside bases
        for _, key in strands.dsbases(self.table):
            delta += self.table[key]
        return delta

    def __dgperbase(self, hpin:Strands, strands:Strands, dtype = 'f8') -> np.ndarray:
        "compute the dg"
        denthalpy = np.zeros(strands.maxsize, dtype = dtype)

        # Terminal endings
        for i, key in strands.terminalbases():
            tmp = self.table.get(key, None)
            if tmp is not None:
                denthalpy[i] += self.dgcor(tmp)

        dgt0 = self.dgcor(self.table['init_A/T'] - self.table['init_G/C'])
        for i, j  in ((strands.dsbegin, 0), (strands.dsend-2, 1)):
            denthalpy[i] += dgt0*(strands.seq[i+j] in 'AaTt')

        # inside bases
        for i, key in strands.dsbases(self.table):
            denthalpy[i] += self.dgcor(self.table[key])

        # We compute salt correction for the hybridized oligo that is with
        # reduced charge near dsDNA
        salt = self.cor*self.saltinfo(hpin, strands, dtype)[0]
        denthalpy[max(0, strands.dsbegin-1):strands.dsend-1] += salt
        return denthalpy

class TransitionMatrix:
    "compute a transition matrix for complex states"
    def __init__(self, *seqs, **_):
        self.dtype   = _DTYPE
        self.strands = StrandList(*seqs)
        self.data    = EnergyComputations(**_)

    def transitionmatrix(self, *masked, states : Optional[np.ndarray] = None) -> np.ndarray:
        """
        all the different states's
        """
        masked, states = self.strands.parseargs(masked, states)
        trans          = np.zeros((len(states),)*2, dtype = self.dtype)
        if self.strands.hpin not in masked:
            ioli = 2
            roo  = self.data.roo(self.strands.hpin, self.strands.hpin, self.dtype)
            rco  = self.data.rch(self.dtype)
            self.__addstates(self.strands.hairpintransitions(states), roo, rco, trans)
        else:
            ioli = 1

        maxs = self.strands.ndigits
        rco  = self.data.rco(self.dtype)
        for oli in self.strands.oligos:
            if oli in masked:
                continue

            roo = self.data.roo(self.strands.hpin, oli, self.dtype)
            for side in (True, False):
                self.__addstates(
                    self.strands.oligotransitions(maxs, states, side, ioli),
                    roo,
                    rco,
                    trans
                )
            ioli += 1

        return trans

    @staticmethod
    def __addstates(states, roo, rco, trans):
        for inds in states:
            lst = list(inds[0])
            if inds['canclose']:
                trans[lst, lst[1]] += rco
            if inds['canopen']:
                trans[lst, lst[0]] += roo[inds['base'], inds['islast'], :]

_DOC = __doc__
class TransitionStats(TransitionMatrix):
    """
    Compute transition stats for complex states
    """
    if __doc__:
        __doc__ = _DOC
    def initialstate(
            self,
            *masked,
            states: Optional[np.ndarray] = None,
            method = InitialState.stable.name
    ):
        """
        Return the initial state.

        This is defined as:

        1. All oligos are bound by at least 2 base.
        2. The probability of such states are determined considering
        transitions without the hairpin moving in.
        3. The hairpin is at the min position for each potential state.
        """
        masked, states = self.strands.parseargs(masked, states)
        return (
            self.__initialstate_full(states, 1) if self.strands.hpin in masked else
            self.__initialstate_full(states, 2) if InitialState(method).name == 'all' else
            self.__initialstate_withhpin_stable(masked, states)
        )

    def compute(
            self,
            *masked,
            ini:         Optional[np.ndarray] = None,
            available:   Optional[np.ndarray] = None,
            states:      Optional[np.ndarray] = None,
            transitions: Optional[np.ndarray] = None
    ):
        "compose a final with an initial state"
        masked, states = self.strands.parseargs(masked, states)
        get          = lambda x, y, **z: (
            x(*masked, states = states, **z)
            if not isinstance(y, (list,np.ndarray, tuple)) else
            y
        )
        transitions  = get(self.transitionmatrix, transitions)
        ini          = get(self.initialstate,     ini, method = ini)
        available    = get(self.strands.transiantstates,  available)

        if available is not None:
            transitions = transitions[available,:][:,available]
            ini         = ini[available]

        out = -_solve(transitions, ini)

        if available is not None:
            tmp, out       = out, np.zeros(len(available), dtype = self.dtype)
            out[available] = tmp
        return out

    def statistics(self, *masked, rate = None, **kwa):
        "all results"
        masked, states = self.strands.parseargs(masked, None)
        available      = self.strands.transiantstates(*masked, states = states)

        vect           = (
            self.compute(*masked, states = states, available = available, **kwa)
            * (rate if rate is not None else 1e-6 if self.strands.hpin in masked else 2.8e-6)
        )

        def _info(key:int, strands: Strands) -> Tuple[float, float, float]:
            good      = states[available, key+1+ (self.strands.hpin not in masked)] != _OFF
            cnt       = vect[available][good].sum()
            temp, dgf = self.data.temperaturesanddgf(self.strands.hpin, strands, self.dtype)
            return temp, dgf, cnt

        info = np.array([
            _info(i, j)
            for i, j in enumerate(k for k in self.strands.oligos if k not in masked)
        ])
        dgf =  np.average(info[:,1], weights = info[:,2])
        return (
            vect.sum(),
            1/vect.sum(),                   # kon
            1/(vect.sum()*np.exp(dgf) * 1000000), # koff
            dgf,
            np.average(info[:,0], weights = info[:,2])
        )

    def __initialstate_full(self, states, first):
        maxs  = self.strands.ndigits
        valid = states[:,first:][states[:,first:].max(1) != _OFF]
        inds  = (valid.min(0) // maxs) * maxs + (valid.max(0) % maxs)
        if first == 2:
            inds = np.insert(inds, 0, max(0, (inds//maxs).min()-1))
        return ((states[:,1:] == inds).sum(1) == (states.shape[1]-1)).astype(self.dtype)

    __INIT_LOOP = 3
    def __initialstate_withhpin_stable(self, masked, states):
        olmask             = (self.strands.hpin,)+masked
        olstates           = self.strands.parseargs(olmask, None)[1]
        left, right        = _frombuffer(*(np.copy(i) for i in (states[:,1:], olstates)))
        maxv               = olstates[:,1:].min(1).ravel()
        maxv[maxv == _OFF] = states[:,1].max()

        right['f0']                   = np.maximum(maxv//self.strands.ndigits, 0)
        right['f0'][right['f0'] == 1] = 0

        inds               = np.searchsorted(left, right)
        good               = left[inds] == right
        stable             = None
        for _ in range(self.__INIT_LOOP):
            stable = self.compute(*olmask, states = olstates, ini = stable)
            stable[(~good) | (stable < 0.)] = 0.
            cnt = stable.sum()
            if cnt <= 0.:
                break
            stable /= cnt

        out       = np.zeros(len(states), dtype = self.dtype)
        out[inds] = stable
        return out
