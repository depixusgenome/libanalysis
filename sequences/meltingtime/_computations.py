#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"Computing melting times"
from itertools  import chain, product
from typing     import Iterator, Tuple, Optional, List, Callable, cast
import numpy  as np
from   numpy.linalg import solve as _solve
from   ._base       import BaseComputations, complement, gccontent, SaltInfo

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
        self.dsend    : int   = len(self.seq)-dsfcn(self.seq[::-1], self.opposite[::-1])

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

    def dskey(self, ind:int) -> str:
        "get table keys"
        if ind == -1:
            return self.dsseq[-2:][::-1]  + '/' + self.dsopposite[-2:][::-1]
        return self.dsopposite[ind:ind+2] + '/' + self.dsseq[ind:ind+2]

    def key(self, ind:int) -> str:
        "get table keys"
        return self.opposite[ind:ind+2] + '/' + self.seq[ind:ind+2]

    def terminalbases(self) -> Iterator[Tuple[int, str]]:
        "return keys for terminal bases"
        if self.dsend < len(self.seq):
            yield (
                self.dsend-1,
                self.opposite[self.dsend-1:self.dsend+1]
                + './'
                + self.seq[self.dsend-1:self.dsend+1]
                + '.'
            )
        if self.dsbegin > 0:
            yield (
                self.dsbegin-1,
                self.seq[self.dsbegin-1:self.dsbegin+1][::-1]
                + './'
                + self.opposite[self.dsbegin-1:self.dsbegin+1][::-1]
                + '.'
            )

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

    def dsbases(self) -> Iterator[Tuple[int, str]]:
        "return keys for the double stranded part of the dna"
        yield from ((i, self.key(i)) for i in range(self.dsbegin, self.dsend-1))

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
        cpy.shift    = (self.dsend - len(self.seq)) * (1 if self.shift > 0 else -1)
        cpy.dsbegin  = len(self.seq)-self.dsend
        cpy.dsend    = len(self.seq)-self.dsbegin
        return cpy

    def nstates(self, nzipped = -1) -> int:
        """
        The number of different states provided *both* ends can de-hybredize.

        The argument *nzipped* tells how many bases are zipped starting from the
        left: this is the effect of the fork.
        """
        assert self.dsbegin > 1 and self.dsend < len(self.seq)
        nds = max(0, self.dsend - max(nzipped+1, self.dsbegin))
        return (nds*(nds-1))//2+1

_OFF: int = cast(int, np.iinfo('i4').max)
class StatesTransitions(BaseComputations):
    "compute results for complex states"
    def __init__(self, hpin, *oligos, chpin = None, **_):
        super().__init__(**_)
        self.dtype  = np.dtype('f8')
        self.hpin   = Strands(hpin, opposite = chpin)
        self.oligos = sorted(
            [self.newstrands(*oligos[i:i+2]) for i in range(0, len(oligos), 2)],
            key = lambda x: x.dsbegin
        )

        test = lambda ols: np.all(
            np.array([i.dsend for i in ols[:-1]]) < np.array([i.dbegin for i in ols[1:]])
        )
        assert test([i for i in self.oligos if i.isoligo])
        assert test([i for i in self.oligos if i.iscoligo])

    def newstrands(self, oligo:str, shift:int) -> Strands:
        "create a new strand"
        return (
            Strands(seq = self.hpin.seq, opposite = oligo,              shift = shift)
            if shift <= 0 else
            Strands(seq = oligo,         opposite = self.hpin.opposite, shift = shift)
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
        rco            = self.__rco()
        hashp          = self.hpin not in masked
        maxs           = self.__maxs

        if hashp:
            roo = self.__roo(self.hpin)
            for inds in self.__join_hp(states):
                trans[inds[0], list(inds[0:2])] += roo[inds[2], inds[3], :]
                trans[inds[1], list(inds[0:2])] += rco

        for (ioli, roo), side in product(
                enumerate(self.__roo(oli) for oli in self.oligos if oli not in masked),
                (False, True)
        ):
            for inds in self.__join_oligo(maxs, states, side, ioli+hashp+1):
                trans[inds[0], list(inds[0:2])] += roo[inds[2], inds[3], :]
                trans[inds[1], list(inds[0:2])] += rco

        return states, trans

    def initialstate(self, *masked, states: Optional[np.ndarray] = None):
        """
        Return the initial state.

        This is defined as:

        1. All oligos are bound by at least 2 base.
        2. The probability of such states are determined considering
        transitions without the hairpin moving in.
        3. The hairpin is at the min position for each potential state.
        """
        masked, states = self.__masked_and_states(masked, states)
        olstates       = self.states(self.hpin, *masked)
        good           = np.max(olstates, axis = 1) != _OFF

        prob           = _solve(
            self.transitions(self.hpin, *masked, states = olstates)[1],
            good.astype(self.dtype)
        )
        prob[np.logical_not(good)] = 0.

        if np.all(prob <= 0.):
            return states, np.zeros(len(states), dtype = self.dtype)

        prob     /= np.sum(prob)

        inds      = self.__find_hpstates(states, olstates)
        out       = np.zeros(len(states), dtype = self.dtype)
        out[inds] = prob
        return states, out

    def finalstate(self, *_ , states: np.ndarray):
        """
        Return the final state.

        This is defined as:

        1. All oligos are off.
        2. The hairpin is fully closed
        """
        out = np.zeros(len(states), dtype = self.dtype)
        out[states['hp'] == len(self.hpin.seq)-1] = 1.
        return out

    def compute(
            self,
            *masked,
            ini:         Optional[np.ndarray] = None,
            fin:         Optional[np.ndarray] = None,
            states:      Optional[np.ndarray] = None,
            transitions: Optional[np.ndarray] = None
    ):
        "compose a final with an initial state"
        mask, states = self.__masked_and_states(masked, states)
        get          = lambda x, y: x(*mask, states = states) if y is None else y
        return (
            _solve(
                get(self.transitions, transitions),
                get(self.initialstate, ini)
            ) @ (
                get(self.finalstate, fin)
            )
        )

    def statistics(self, *masked, rate = 1e-6*2.8):
        "all results"
        trep        = rate*self.compute(*masked)
        temp, delta = self.__saltinfo(self.hpin)[2:]
        dgf         = self.cor*delta+((len(self.hpin)-1)*(self.gss-self.gds))
        return (
            trep,
            1/trep,                         # kon
            1/(trep*np.exp(dgf) * 1000000), # koff
            dgf,
            temp
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
        for _, key in strands.dsbases():
            delta += self.table[key]
        return delta

    def __dgperbase(self, strands:Strands) -> np.ndarray:
        "compute the dg"
        denthalpy = np.zeros(len(strands.seq), dtype = self.dtype)

        # Terminal endings
        for i, key in strands.terminalbases():
            tmp = self.table.get(key, None)
            if tmp is not None:
                denthalpy[i] += tmp

        dgt0 = self.dgcor(self.table['init_A/T'] - self.table['init_G/C'])
        for i in (strands.dsbegin, strands.dsend-1):
            denthalpy[i] += dgt0*(strands.seq[i] in 'AaTt')

        # inside bases
        for i, key in strands.dsbases():
            denthalpy[i] += self.dgcor(self.table[key])

        # We compute salt correction for the hybridized oligo that is with
        # reduced charge near dsDNA
        salt = self.cor*self.__saltinfo(strands)[0]
        denthalpy[max(0, strands.dsbegin-1):strands.dsend-1] += salt
        return denthalpy

    def __roo(self, strands:Strands):
        "create the roo vector"
        sli        = slice(strands.dsbegin, strands.dsend-1)

        roo        = np.zeros((len(strands.seq),2), dtype = self.dtype)
        roo[sli,0] = -self.__dgperbase(strands)[sli]

        dgt0       = self.dgcor(self.table['init_A/T'] - self.table['init_G/C'])
        if strands.seq[cast(int, sli.start)] in 'AaTt':
            roo[sli.start,0]             -= -dgt0
        if strands.seq[cast(int, sli.stop)] in 'AaTt':
            roo[cast(int, sli.stop)-1,0] += dgt0

        roo[sli,1] += roo[sli,0]-self.dgcor(self.table['init_G/C'])
        roo[sli,:]  = np.exp(roo[sli,:])
        return np.kron(roo, [-1., 1.]).reshape((-1, 2, 2))

    def __rco(self):
        "return closing factors"
        return self.elasticity.rco(self.temperatures)*np.array([1, -1], dtype = self.dtype)

    @staticmethod
    def __join(
            states:  np.ndarray,
            aopened: np.ndarray,
            ison:    np.ndarray,
            keys:    List[int]
    )-> Tuple[np.ndarray, np.ndarray]:
        closed, opened  = [
            np.recarray(buf.shape[:1], formats = ['i4']*buf.shape[1], buf = buf)
            for buf in (np.copy(states[:,1:]), aopened)
        ]

        inds        = np.searchsorted(closed, opened)
        good        = closed[inds] == opened
        inds        = inds[good]

        out        = states[ison, :][:, keys][good]
        out[:,1]   = states[inds, :][:,0]
        return inds, out

    @classmethod
    def __join_hp(cls, states) -> np.ndarray:
        ison                          = states[:,1] > 0
        aopened                       = states[:,1:][ison]
        aopened[:,0]                 -= 1
        aopened[aopened[:,0] == 1, 0] = 0 # require 2 bases to hold the hp

        inds, out = cls.__join(states, aopened, ison, [0, 0, 1, 1])
        out[:,2] -= 2
        out[:,3]  = states[inds,1] == 0
        return out

    @classmethod
    def __join_oligo(cls, maxs:int, states, isright:bool, key:int) -> np.ndarray:
        off                  = states[:,key] != _OFF
        aopened              = states[:,1:][off]
        aopened[:,key-1]    += -1 if isright else maxs
        going                = aopened[:,key-1]//maxs+1 >= (aopened[:,key-1]%maxs)
        aopened[going,key-1] = _OFF

        inds, out            = cls.__join(states, aopened, off, [0, 0, key, key])

        out[:,2] = ((out[:,2] % maxs)-2) if isright else (out[:,2] // maxs)
        out[:,3] = states[inds,key] == _OFF
        return out

    @staticmethod
    def __find_hpstates(states, olstates):
        aclosed        = np.copy(states[:,1])
        aopened        = np.copy(aclosed)
        aopened[:,0]   = np.max(olstates[:,1:], axis = 1)-1

        closed, opened = [
            np.recarray(buf.shape[:1], formats = ['i4']*buf.shape[1], buf = buf)
            for buf in (aclosed, aopened)
        ]

        inds = np.clip(np.searchsorted(closed, opened), 0, len(closed)-1)
        return states[:,0][inds[closed[inds] == opened]]

    @property
    def __maxs(self) -> int:
        return 10**(int(np.round(np.log10(len(self.hpin.seq))))+1)

    def __masked_and_states(
            self,
            masked: tuple,
            states: Optional[np.ndarray]
    ) -> Tuple[Tuple[Strands,...], np.ndarray]:
        if states is None:
            states = next((i for i in masked if isinstance(i, np.ndarray)), None)
            masked = tuple(i for i in masked if not isinstance(i, np.ndarray))

        if states is None:
            states = self.states(*masked)
        return masked, states
