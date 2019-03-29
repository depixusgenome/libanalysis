#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"Computing melting times"
from   enum          import Enum
from   itertools     import chain, product
from   typing        import Iterator, Tuple, Optional, List, Callable, cast
import numpy  as np
from   numpy.linalg  import solve as _solve
from   ._base        import BaseComputations, complement, gccontent, SaltInfo

class InitialState(Enum):
    "Compuptation mode of the initial state"
    stable     = "stable"
    hybridized = "hybridized"

class Strands:
    "all about strands"
    def __init__(
            self,
            seq:        Optional[str],
            opposite:   Optional[str] = None,
            shift:      int           = 0,
            density:    float         = 25
    ):
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

_JOIN_DT: np.dtype = np.dtype([
    ('states',  np.dtype([('iclose',  'i4'), ('iopen',    'i4')])),
    ('base',     'i4'),
    ('islast',   'i1'),
    ('canclose', 'bool'),
    ('canopen',  'bool'),
    ('_',        'i1')
])

_OFF:     int      = cast(int, np.iinfo('i4').max)
class StatesTransitions(BaseComputations):
    "compute results for complex states"
    def __init__(self, hpin, *oligos, chpin = None, **_):
        super().__init__(**_)
        self.dtype  = np.dtype('f8')
        self.hpin   = Strands(hpin, opposite = chpin)
        self.oligos = sorted(
            [
                self.newstrands(*i)  if isinstance(i, (list, tuple)) else
                self.newstrands(**i) if isinstance(i, dict) else
                self.newstrands(i)
                for i in oligos
            ],
            key = lambda x: x.dsbegin
        )

        test = lambda ols: np.all(
            np.array([i.dsend for i in ols[:-1]]) < np.array([i.dbegin for i in ols[1:]])
        )
        assert test([i for i in self.oligos if i.isoligo])
        assert test([i for i in self.oligos if i.iscoligo])

    def newstrands(self, oligo:str, seq:bool = False, shift:int = 0) -> Strands:
        "create a new strand"
        return (
            Strands(seq = oligo,         opposite = self.hpin.opposite, shift = shift)
            if seq else
            Strands(seq = self.hpin.seq, opposite = oligo,              shift = shift)
        )

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

    def transitions(self, *masked, states : Optional[np.ndarray] = None) -> np.ndarray:
        """
        all the different states's
        """
        masked, states = self.__masked_and_states(masked, states)
        trans          = np.zeros((len(states),)*2, dtype = self.dtype)
        if self.hpin not in masked:
            ioli = 2
            roo  = self.__roo(self.hpin)
            rco  = self.elasticity.rch(self.temperatures)*np.array([1, -1], dtype = self.dtype)
            self.__addstates(self.__join_hp(states, states[:,1] > 0), roo, rco, trans)
        else:
            ioli = 1

        maxs = self.__maxs
        rco  = self.elasticity.rco(self.temperatures)*np.array([1, -1], dtype = self.dtype)
        for oli in self.oligos:
            if oli in masked:
                continue

            roo = self.__roo(oli)
            for side in (True, False):
                self.__addstates(
                    self.__join_oligo(maxs, states, side, ioli),
                    roo,
                    rco,
                    trans
                )
            ioli += 1

        return trans

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
        masked, states = self.__masked_and_states(masked, states)
        return (
            self.__initialstate_full(states, 1) if self.hpin in masked else
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
        masked, states = self.__masked_and_states(masked, states)
        get          = lambda x, y, **z: (
            x(*masked, states = states, **z)
            if not isinstance(y, (list,np.ndarray, tuple)) else
            y
        )
        transitions  = get(self.transitions,      transitions)
        ini          = get(self.initialstate,     ini, method = ini)
        available    = get(self.transitionstates, available)

        if available is not None:
            transitions = transitions[available,:][:,available]
            ini         = ini[available]

        out = -_solve(transitions, ini)

        if available is not None:
            tmp, out       = out, np.zeros(len(available), dtype = self.dtype)
            out[available] = tmp
        return out

    def transitionstates(self, *masked, states: Optional[np.ndarray] = None) -> np.ndarray:
        "states to be used for computations"
        states = self.__masked_and_states(masked, states)[1]
        if self.hpin in masked:
            return states[:,1:].max(1) != _OFF
        return states[:,1] != states[:,1].max()

    def statistics(self, *masked, rate = None, **kwa):
        "all results"
        masked, states = self.__masked_and_states(masked, None)
        available      = self.transitionstates(*masked, states = states)

        vect           = (
            self.compute(*masked, states = states, available = available, **kwa)
            * (rate if rate is not None else 1e-6 if self.hpin in masked else 2.8e-6)
        )

        def _info(key:int, strands: Strands) -> Tuple[float, float, float]:
            temp, delta = self.__saltinfo(strands)[2:]
            good        = states[available, key+1+ (self.hpin not in masked)] != _OFF
            cnt         = vect[available][good].sum()
            return temp, self.cor*delta+(strands.dssize-1)*(self.gss-self.gds), cnt

        info = np.array([
            _info(i, j)
            for i, j in enumerate(k for k in self.oligos if k not in masked)
        ])
        dgf =  np.average(info[:,1], weights = info[:,2])
        return (
            vect.sum(),
            1/vect.sum(),                   # kon
            1/(vect.sum()*np.exp(dgf) * 1000000), # koff
            dgf,
            np.average(info[:,0], weights = info[:,2])
        )

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
        maxs = self.__maxs
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

    def __saltinfo(self, strands: Strands) -> SaltInfo:
        "return the salt info for a given oligo"
        delta = self.__delta(strands)
        return self.salt.compute(
            strands.dsseq,
            self.hpin.density,
            strands.density,
            self.temperatures.mtG,
            delta
        )

    def __delta(self, strands:Strands) -> np.ndarray:
        "compute the deltas"
        dsseq   = strands.dsseq
        delta    = np.array([0., 0.], dtype = self.dtype)

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

    def __dgperbase(self, strands:Strands) -> np.ndarray:
        "compute the dg"
        denthalpy = np.zeros(strands.maxsize, dtype = self.dtype)

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
        salt = self.cor*self.__saltinfo(strands)[0]
        denthalpy[max(0, strands.dsbegin-1):strands.dsend-1] += salt
        return denthalpy

    def __roo(self, strands:Strands):
        "create the roo vector"
        sli        = slice(strands.dsbegin, strands.dsend-1)

        roo        = np.zeros((strands.maxsize,2), dtype = self.dtype)
        roo[sli,0] = -self.__dgperbase(strands)[sli]

        dgt0       = self.dgcor(self.table['init_A/T'] - self.table['init_G/C'])
        if strands.seq[cast(int, sli.start)] in 'AaTt':
            roo[sli.start,0]             += -dgt0
        if strands.seq[cast(int, sli.stop)] in 'AaTt':
            roo[cast(int, sli.stop)-1,0] += dgt0

        roo[sli,1] += roo[sli,0]-self.dgcor(self.table['init_G/C'])
        roo[sli,:]  = np.exp(roo[sli,:])
        return np.kron(roo, [-1., 1.]).reshape((-1, 2, 2))

    @classmethod
    def __join(
            cls,
            states:  np.ndarray,
            aopened: np.ndarray,
            ison:    np.ndarray,
            key:     int
    ) -> np.ndarray:
        closed, opened = cls.__frombuffer(np.copy(states[:,1:]), aopened)
        inds           = np.searchsorted(closed, opened)
        good           = closed[inds] == opened
        inds           = inds[good]

        keys = [0, 0, key, key]
        out  = np.frombuffer(states[ison, :][:, keys][good], dtype = _JOIN_DT)
        out['states']['iopen']       = states[inds, :][:,0]
        out[['canclose', 'canopen']] = True
        out['islast']                = False
        return out

    @classmethod
    def __join_hp(cls, states, ison) -> np.ndarray:
        aopened                       = states[:,1:][ison]
        aopened[:,0]                 -= 1
        aopened[aopened[:,0] == 1, 0] = 0 # require 2 bases to hold the hp

        out            = cls.__join(states, aopened, ison, 1)
        out['canopen'] = (out['base'] > 2) & (out['base'] < states[:,1].max())
        out['base']    = np.maximum(out['base']-2, 0)
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

    @classmethod
    def __join_oligo(cls, maxs:int, states, isright: bool, key:int) -> np.ndarray:
        return (cls.__join_oligo_right if isright else cls.__join_oligo_left)(maxs, states, key)

    @staticmethod
    def __addstates(states, roo, rco, trans):
        for inds in states:
            lst = list(inds[0])
            if inds['canclose']:
                trans[lst, lst[1]] += rco
            if inds['canopen']:
                trans[lst, lst[0]] += roo[inds['base'], inds['islast'], :]

    def __apriori(self, masked, states, trans):
        """
        Applying aprioris to the transitions.
        """

    @staticmethod
    def __find_hpstates(states, olstates):
        aclosed        = np.copy(states[:,1])
        aopened        = np.copy(aclosed)
        aopened[:,0]   = np.max(olstates[:,1:], axis = 1)-1

        closed, opened = [
            np.frombuffer(buf, dtype = 'i4,'*buf.shape[1])
            for buf in (aclosed, aopened)
        ]

        inds = np.clip(np.searchsorted(closed, opened), 0, len(closed)-1)
        return states[:,0][inds[closed[inds] == opened]]

    @property
    def __maxs(self) -> int:
        return 10**(int(np.round(np.log10(self.hpin.maxsize)))+1)

    def __masked_and_states(
            self,
            masked: tuple,
            states: Optional[np.ndarray]
    ) -> Tuple[Tuple[Strands,...], np.ndarray]:
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

    def __initialstate_full(self, states, first):
        maxs  = self.__maxs
        valid = states[:,first:][states[:,first:] != _OFF]
        inds  = (valid.min(0) // maxs) * maxs + (valid.max(0) % maxs)
        if first == 2:
            inds = np.insert(inds, 0, max(0, (inds//maxs).min()-1))
        return (states[:,1:] == inds).ravel().astype(self.dtype)

    __INIT_LOOP = 3
    def __initialstate_withhpin_stable(self, masked, states):
        olmask             = (self.hpin,)+masked
        olstates           = self.__masked_and_states(olmask, None)[1]
        left, right        = self.__frombuffer(*(np.copy(i) for i in (states[:,1:], olstates)))
        maxv               = olstates[:,1:].min(1).ravel()
        maxv[maxv == _OFF] = states[:,1].max()

        right['f0']                   = np.maximum(maxv//self.__maxs, 0)
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

    @staticmethod
    def __frombuffer(*buffers):
        return (
            np.frombuffer(
                buf,
                dtype = [(f'f{i}', 'i4') for i in range(buf.shape[1])]
            )
            for buf in buffers
        )
