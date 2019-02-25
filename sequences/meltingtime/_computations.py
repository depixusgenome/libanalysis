#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"Computing melting times"
from itertools  import chain, product
from typing     import Iterator, Tuple, Optional, cast
import numpy  as np
from   numpy.lib.recfunctions import join_by
from   ._base import BaseComputations, complement, gccontent

class Strands:
    "all about strands"
    def __init__(self, seq:Optional[str], opposite:Optional[str] = None, shift: int = 0):
        if opposite is None:
            opposite = complement(cast(str, seq))
        elif seq is None:
            seq      = complement(cast(str, opposite))
        assert isinstance(seq,      str)
        assert isinstance(opposite, str)

        seq      = '.' * max(0, -shift) + seq
        opposite = '.' * max(0, shift)  + opposite

        self.seq      = seq    + '.' * max(0, len(opposite) - len(seq))
        self.opposite = opposite  + '.' * max(0, len(seq)   - len(opposite))
        self.shift    = shift

        dsfcn   = lambda x, y: next(i for i, j in enumerate(zip(x, y)) if '.' not in j)
        self.dsbegin = dsfcn(self.seq, self.opposite)
        self.dsend   = len(self.seq)-dsfcn(self.seq[::-1], self.opposite[::-1])

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
        yield (
            self.dsend-1,
            self.opposite[self.dsend-1:self.dsend+1]
            + './'
            + self.seq[self.dsend-1:self.dsend+1]
            + '.'
        )
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

class StatesTransitions(BaseComputations):
    "compute results for complex states"
    def __init__(self, hpin, *oligos, chpin = None, **_):
        super().__init__(**_)
        self.hpin   = Strands(hpin, opposite = chpin)
        self.oligos = sorted(
            [self.newstrands(*oligos[i:i+2]) for i in range(0, len(oligos), 2)],
            key = lambda x: x.dsbegin
        )

        assert np.all(np.diff([i.dsend for i in self.oligos if i.isoligo]) > 0)
        assert np.all(np.diff([i.dsend for i in self.oligos if i.iscoligo]) > 0)

    def newstrands(self, oligo:str, shift:int) -> Strands:
        "create a new strand"
        return (
            Strands(seq = self.hpin.seq, opposite = oligo,              shift = shift)
            if shift < 0 else
            Strands(seq = oligo,         opposite = self.hpin.opposite, shift = shift)
        )

    def nstates(self) -> np.ndarray:
        """
        number of different states
        """
        inds = np.array([(i.dsbegin, i.dsend) for i in self.oligos], dtype = 'i4')

        out  = 1 # the ∅ state
        for i in range(len(self.hpin.seq)):
            # hairpin is now zipped up to *i*
            nvals  = cast(np.ndarray, np.maximum(inds[:,1]-np.maximum(inds[:,0], i+1), 0))
            out   += np.prod((nvals*(nvals-1))//2 + 1)
        return out

    def states(self) -> np.recarray:
        """
        all the different states
        """
        inds  = np.array([(i.dsbegin, i.dsend) for i in self.oligos], dtype = 'i4')
        maxs  = int(np.round(np.log10(len(self.hpin.seq))))+1
        assert maxs^2 < np.iinfo('u4').max()

        inner = [
            (
                np.insert(
                    np.cumsum(list(range(i.dsend-i.dsbegin-1))[::-1], dtype = 'i4'),
                    0, 0
                ),
                np.append(
                    np.array(
                        chain(
                            range((j-1)*maxs+j+1, (j-1)*maxs+i.dsend+1)
                            for j in range(i.dsbegin+1, i.dsend)
                        ),
                        dtype = 'u4'
                    ),
                    0, 0
                ),
                i.dsbegin
            ) for i in self.oligos
        ]

        prev = cur = 0
        outs = np.zeros((self.nstates(), 2 + len(inds)), dtype = 'f4')
        for i in range(len(self.hpin.seq)):
            nvals     = cast(np.ndarray, np.maximum(inds[:,1]-np.maximum(inds[:,0], i+1), 0))
            cur      += np.prod((nvals*(nvals-1))//2 + 1)
            outs[prev:cur, 1]  = i+1
            outs[prev:cur, 2:] = list(
                product(*(k[j[max(i+1, l)]:,:] for j, k, l in inner))
            )
        outs[:,0] = np.arange(outs.shape[0], dtype = 'i4')

        cols   = ["state", "hp", *(f"o{i}" for i in range(len(self.oligos)))]
        return np.recarray(
            (outs.size[0],),
            names   = cols,
            formats = ('i4',)*len(cols),
            buf     = outs
        )

    def transitions(self) -> np.ndarray:
        """
        all the different states's
        """
        states = self.states()
        maxs   = int(np.round(np.log10(len(self.hpin.seq))))+1
        trans  = np.zeros((len(states),)*2, dtype = 'f4')

        self.__hairpintransitions(states, trans)


        for i in range(len(self.oligos)):
            self.__oligotransitions(i, maxs, states, trans)

    def rco(self):
        "return closing factors"
        return -self.elasticity.rco(self.temperatures)*np.array([-1, 1], dtype = 'f4')

    def roo(self, strands:Strands):
        "create the roo vector"
        dgpb = self.dgperbase(strands)

        dgt0 = self.dgcor(self.table['init_A/T'] - self.table['init_G/C'])
        dgt  = self.dgcor(self.table['init_G/C'])

        roo         = np.zeros((len(dgpb)+1,2), dtype = 'f4')
        roo[:-1,0]  = np.exp(-dgpb)
        roo[:-1,1]  = roo[:-1,0]
        roo[:-1,1] *= np.exp(-dgt)
        if strands.seq[strands.dsbegin] in 'AaTt':
            roo[0,:]  = np.exp(-dgpb[0]-dgt0), np.exp(-dgpb[0]-dgt0-dgt)
        if strands.seq[strands.dsend-1] in 'AaTt':
            roo[-2,:] = np.exp(-dgpb[-1]+dgt0), np.exp(-dgpb[-1]+dgt0-dgt)
        return np.kron(roo, [-1., 1.]).reshape((-1, 2, 2))

    def delta(self, strands:Strands) -> np.ndarray:
        "compute the deltas"
        dsseq   = strands.dsseq
        delta    = np.array([0., 0.], dtype = 'f8')

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

    def dgperbase(self, strands:Strands) -> np.ndarray:
        "compute the dg"
        denthalpy = np.zeros(len(strands.seq), dtype = 'f8')

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
        return denthalpy

    def __hairpintransitions(self, states, trans):
        rco         = self.rco()
        roo         = self.roo(self.hpin)[:,0]
        other       = np.copy(states)
        other.hpin += 1
        joins       = list(states.dtype.names[1:])
        for j in join_by(joins, states, other, usemask = False)[["state1", "state2"]]:
            trans[j[0], j] += roo[j[0], :]
            trans[j[1], j] += rco

    def __oligotransitions(self, ioli, maxs, states, trans):
        rco   = self.rco()
        roo   = self.roo(self.oligos[ioli])
        name  = f"o{ioli}"
        joins = list(states.dtype.names[1:])

        other                               = np.copy(states)
        ind                                 = other[name]
        ind[:]                             += 1
        ind[(ind % maxs)  >= (ind // maxs)] = 0
        for i, j, k in join_by(joins, states, other, usemask = False)[[name, "state1", "state2"]]:
            trans[j, [j, k]] += roo[j, i//maxs-(i%maxs) < 2, :]
            trans[k, [j, k]] += rco

        ind[:]                              = states[name]-maxs
        ind[(ind % maxs)  >= (ind // maxs)] = 0
        for i, j, k in join_by(joins, states, other, usemask = False)[[name, "state1", "state2"]]:
            trans[j, [j, k]] += roo[j, i//maxs-(i%maxs) < 2, :]
            trans[k, [j, k]] += rco
