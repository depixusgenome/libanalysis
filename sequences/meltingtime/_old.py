#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"Computing melting times"
from itertools  import chain
from typing     import Iterator, Callable, cast

import numpy  as np

from ._base     import BaseComputations,  gccontent, complement

class KeyComputer:
    "computes keys for various energy tables"
    def __init__(self, seq, oligo):
        self.seq   = str(seq)
        self.oligo = str(oligo if oligo else complement(seq))

    def key(self, ind:int) -> str:
        "get table keys"
        if ind == -1:
            return self.seq[-2:][::-1]  + '/' + self.oligo[-2:][::-1]
        return self.oligo[ind:ind+2] + '/' + self.seq[ind:ind+2]

    def terminalkey(self, ind) -> str:
        "get table keys for terminal endings"
        if ind == -1:
            return self.oligo[-2:] + './' + self.seq[-2:]        + '.'
        return self.seq[:2][::-1]  + './' + self.oligo[:2][::-1] + '.'

    def pop(self, ind):
        "shorten the current sequences"
        if ind == 0:
            self.seq   = self.seq[1:]
            self.oligo = self.oligo[1:]
        elif ind == -1:
            self.seq   = self.seq[:-1]
            self.oligo = self.oligo[:-1]

class ComputationDetails(KeyComputer): # pylint: disable=too-many-instance-attributes
    "All info for computing the melting times & other stats"
    dg:  np.ndarray
    dgh: np.ndarray
    def __init__(self, seq, oligo):
        super().__init__(seq, oligo)
        self.oseq   = self.seq
        self.ooligo = self.oligo
        self.delta  = np.zeros(2, dtype = 'f8')

    def resetdg(self):
        "sets dg & dgh"
        self.dg    = np.zeros(len(self.oseq)-1, dtype = 'f8')
        self.dgh   = np.zeros(len(self.oseq)-1, dtype = 'f8')

class OldStatesTransitions(BaseComputations): # pylint: disable=too-many-instance-attributes
    """
    Return the Tm using nearest neighbor thermodynamics.

    Arguments:
     - sequence: The primer/probe sequence as string or Biopython sequence object.
       For RNA/DNA hybridizations sequence must be the RNA sequence.
     - oligo: Complementary sequence. The sequence of the template/target in
       3'->5' direction. oligo is necessary for mismatch correction and
       dangling-ends correction. Both corrections will automatically be
       applied if mismatches or dangling ends are present. Default=None. It is also
       needed to define the existance of DNA(lower cap) LNA (upper cap) or RNA (???).
     - shift: Shift of the primer/probe sequence on the template/target
       sequence, e.g.::

                           shift=0       shift=1        shift= -1
        Primer (sequence): 5' ATGC...    5'  ATGC...    5' ATGC...
        Template (oligo):  3' TACG...    3' CTACG...    3'  ACG...

       The shift parameter is necessary to align sequence and oligo if they have
       different lengths or if they should have dangling ends. Default=0
     - terminal: Thermodynamic values for terminal mismatches.
       Default: DNA_TMM1 (SantaLucia & Peyret, 2001)
     - table: Thermodynamic NN (1), missmatch (2) and dangling ends(3):
        1. Thermodynamic NN values, eight tables are implemented:
            * For DNA/DNA hybridizations:

                - DNA_NN1: values from Breslauer et al. (1986)
                - DNA_NN2: values from Sugimoto et al. (1996)
                - DNA_NN3: values from Allawi & SantaLucia (1997) (default)
                - DNA_NN4: values from SantaLucia & Hicks (2004)

            * For RNA/RNA hybridizations:

                - RNA_NN1: values from Freier et al. (1986)
                - RNA_NN2: values from Xia et al. (1998)
                - RNA_NN3: valuse from Chen et al. (2012)

            * For RNA/DNA hybridizations:

                - R_DNA_NN1: values from Sugimoto et al. (1995)

           Use the module's maketable method to make a new table or to update one
           one of the implemented tables.

        2. Thermodynamic values for internal mismatches, may include insosine
           mismatches. Default: DNA_IMM1 (Allawi & SantaLucia, 1997-1998 Peyret et
           al., 1999; Watkins & SantaLucia, 2005)
        3. dangling: Thermodynamic values for dangling ends:
            - DNA_DE1: for DNA. Values from Bommarito et al. (2000). Default
            - RNA_DE1: for RNA. Values from Turner & Mathews (2010)

     - seqconcentration: Concentration of the higher concentrated strand [nM]. Typically
       this will be the primer (for PCR) or the probe. Default=25.
     - oligoconcentration: Concentration of the lower concentrated strand [nM]. In PCR this
       is the template strand which concentration is typically very low and may
       be ignored (oligoconcentration=0). In oligo/oligo hybridization experiments, seqconcentration
       equals seqconcentration. Default=25.
       MELTING and Primer3Plus use k = [Oligo(Total)]/4 by default. To mimic
       this behaviour, you have to divide [Oligo(Total)] by 2 and assign this
       concentration to seqconcentration and oligoconcentration. E.g., Total oligo concentration of
       50 nM in Primer3Plus means seqconcentration=25, oligoconcentration=25.
       the primer is thought binding to itself, thus oligoconcentration is not considered.
     - conc_Na, conc_K, conc_Tris, Mg, dNTPs: See method 'Tm_GC' for details. Defaults: conc_Na=50,
       conc_K=0, conc_Tris=0, Mg=0, dNTPs=0.
     - saltcorr: See method 'Tm_GC'. Default=5. 0 means no salt correction.
    """
    def __init__(self, **_):
        super().__init__(**_)
        self.rate         : float             = 1.e-6
        self.fork_tau_mul : float             = 2.8
        self.loop         : bool              = False
        for i in set(self.__dict__) & set(_):
            setattr(self, i, _[i])

    def setmode(self, mode:str):
        "sets the computation mode"
        super().setmode(mode)
        self.loop = mode != 'fork'
        if mode == 'fork':
            self.fork_tau_mul     = 2.8
        return self

    def energies(
            self,
            comp,
            shift    : int = 0,
            rhoseq   : int = 25,
            rhooligo : int = 25
    ):
        "computes the δ enthalpies & entropies"
        self.__shift(shift, comp) # compute resized sequences, withouth overdangling ends
        comp.resetdg()            # make sure dg & dgh have updated sizes

        # Now for terminal mismatches
        off_bp = self.__terminal_mismatches(comp)

        # Now everything 'unusual' at the ends is handled and removed and we can
        # look at the initiation.
        # One or several of the following initiation types may apply:

        # Type: General initiation value
        comp.delta += self.table['init']

        # Type: Duplex with no (allA/T) or at least one (oneG/C) GC pair
        comp.delta += self.table['init_'+('oneG/C' if gccontent(comp.oseq.upper()) else 'allA/T')]

        # Type: Penalty if 5' end is T
        comp.delta += self.table['init_5T/A'] * ((comp.oseq[0] in 'Tt')+(comp.oseq[-1] in 'Aa'))

        # Type: Different values for G/C or A/T terminal basepairs
        ends = (comp.oseq[0] + comp.oseq[-1]).upper()
        for tpe in ("AT", "GC"):
            tmp         = self.table[f'init_{tpe[0]}/{tpe[1]}']
            comp.delta += sum(ends.count(i) for i in tpe) * np.array(tmp)

        dgt0         = self.dgcor(self.table['init_A/T'] - self.table['init_G/C'])
        comp.dg[0]  += dgt0*(comp.oseq[0]  in 'AaTt')
        comp.dg[-1] += dgt0*(comp.oseq[-1] in 'AaTt')

        # Finally, the 'zipping'
        self.__zipping((comp.seq,  comp.oligo), comp.dg[off_bp:], comp.delta)
        if not self.loop:
            self.__zipping((comp.oseq, complement(comp.oseq)),  comp.dgh, None)

        # We compute salt correction for the hybridized oligo that is with
        # reduced charge near dsDNA
        saltinfo  = self.salt.compute(comp.oseq,
                                      rhoseq, rhooligo, self.temperatures.mtG,
                                      comp.delta)
        comp.dg  += self.cor*saltinfo[0]
        comp.dgh += self.cor*saltinfo[0]

    def states(
            self,
            comp,
            rhoseq   : int = 25,
            rhooligo : int = 25
    ):
        "return the states matrix"
        # roo[0] opening of the first 5' base of the oligo towards the fork
        # l is an index over all possible configurations
        #ind[i, j]  is the index of the configuration :
        # i = position of the opening front on the 5' end, [0, nn-1]
        # j = position of the opening front on the 3' end, nn-1 nothing open
        # loop version
        nbases, inds = self.__state_indexes(comp)
        mat          = self.__transitions(nbases, inds, comp)
        if self.loop:
            saltinfo  = self.salt.compute(comp.oseq,
                                          rhoseq, rhooligo, self.temperatures.mtG,
                                          comp.delta)
            self.__encircling(saltinfo[1], inds, mat)
        else:
            self.__fork(comp, inds, mat)
        return mat, inds

    def __call__( # pylint: disable=too-many-arguments
            self,
            sequence : str, # 3'-> 5'
            oligo    : str, # 5'-> 3'
            shift    : int = 0,
            rhoseq   : int = 25,
            rhooligo : int = 25
    ):
        comp        = ComputationDetails(sequence, oligo)
        self.energies(comp, shift, rhoseq, rhooligo)
        trep = self.__trep(comp, *self.states(comp, rhoseq, rhooligo))

        temp, delta = self.salt.compute(comp.oseq,
                                        rhoseq, rhooligo, self.temperatures.mtG,
                                        comp.delta)[2:]
        dgf         = self.cor*delta+((len(comp.oseq)-1)*(self.gss-self.gds))
        return (
            trep,
            1/trep,                         # kon
            1/(trep*np.exp(dgf) * 1000000), # koff
            dgf,
            temp
        )

    def __terminal_mismatches(self, comp) -> bool:
        for i in (0, -1):
            val = self.table.get(comp.terminalkey(i), None)
            if val is not None:
                comp.delta += val
                comp.dg[i] += self.dgcor(val)
                comp.pop(i)
        return comp.oseq[0] != comp.seq[0]

    def __trep(self, comp, mat, inds):
        #initial state: nn open bases for the hairpin 0 open bases for the oligo
        ini                                                     = np.zeros(mat.shape[0])
        ini[inds[(0,)*(len(inds.shape)-1)+(len(comp.oseq)-1,)]] = 1

        #holding state: all possible configurations
        hold                                                    = np.ones_like(ini)

        # between the initial and final state
        trep =-self.rate*(np.linalg.solve(mat, ini) @ hold) # type: ignore
        if not self.loop:
            trep *= self.fork_tau_mul
        return trep

    def __shift(self, shift:int, comp:ComputationDetails):
        # Dangling ends?
        if shift or len(comp.seq) != len(comp.oligo):
            # Align both sequences using the shift parameter
            bigger      = len(comp.seq) > len(comp.oligo)
            smaller     = len(comp.seq) < len(comp.oligo)
            comp.seq    = '.' * (shift < 0 or smaller) + comp.seq[max(shift-1,  0):]
            comp.oligo  = '.' * (shift > 0 or bigger)  + comp.oligo[max(-shift-1, 0):]

            bigger      = len(comp.seq) > len(comp.oligo)
            smaller     = len(comp.seq) < len(comp.oligo)
            comp.seq    = comp.seq[:len(comp.oligo)+bigger]  + '.' * smaller
            comp.oligo  = comp.oligo[:len(comp.seq)+smaller] + '.' * bigger

            # Now for the dangling ends
            for ind in (0, -1):
                if any(i[ind] == "." for i in (comp.seq, comp.oligo)):
                    comp.delta += self.table.get(comp.key(ind), 0.)
                    comp.pop(ind)
            comp.oseq   = comp.seq
            comp.ooligo = comp.oligo
            if len(comp.seq) <= 1 or len(comp.oligo) <= 1:
                raise NotImplementedError("Don't deal with single base oligos")

    def __state_indexes(self, comp):
        nbases = len(comp.oseq)
        ind    = np.zeros((nbases,)*(3-self.loop), dtype='i8')
        cnt    = 0
        if self.loop:
            itr: Callable[[int], Iterator] = lambda i: iter(((i,),))
        else:
            itr = lambda i: ((i, j) for j in range(i+1))

        for i in range(nbases):
            for j in cast(Iterator, itr(i)):
                ind[j+(slice(i+1, nbases),)] = np.arange(nbases-i-1) + cnt
                cnt    += nbases-i-1

        return nbases, ind

    @staticmethod
    def __iter5prime(inds):
        nbases = inds.shape[0]
        if len(inds.shape) == 2:
            yield from (
                (inds[i,j], inds[i+1,j], i, j-i > 2)
                for j in range(2,nbases) for i in range(j-1)
            )
            return

        yield from (
            (inds[i,j,k], inds[i+1,j, k], i, k-i > 2)
            for j in range(nbases-1) for i in range(j,nbases-2) for k in range (i+2,nbases)
        )

    @staticmethod
    def __iter3prime(inds):
        nbases = inds.shape[0]
        if len(inds.shape) == 2:
            yield from (
                (inds[i,j], inds[i,j-1], j-1, j-i > 2)
                for i in range(nbases) for j in range(i+2, nbases)
            )
            return
        yield from (
            (inds[i,j,k], inds[i,j,k-1], k-1, k-i > 2)
            for i in range(nbases) for j in range(i+1) for k in range(i+2, nbases)
        )

    @staticmethod
    def __iterescaping(nbases, inds):
        if len(inds.shape) == 2:
            yield from ((inds[i,i+1], i) for i in range(nbases-1))
            return

        yield from ((inds[i,j,i+1], i) for i in range (nbases-1) for j in range(i+1))

    def __roo(self, comp):
        dgt0 = self.dgcor(self.table['init_A/T'] - self.table['init_G/C'])
        roo  = np.append(np.exp(-comp.dg),     0)
        if comp.seq[0] in 'AaTt':
            roo[0]  = np.exp(-comp.dg[0]-dgt0)
        if comp.seq[-1] in 'AaTt':
            roo[-2] = np.exp(-comp.dg[-1]+dgt0)
        return roo

    def __roo2(self, comp):
        dgt0 = self.dgcor(self.table['init_A/T'] - self.table['init_G/C'])
        dgt  = self.dgcor(self.table['init_G/C'])
        roo2 = np.append(np.exp(-comp.dg-dgt), 0)
        if comp.seq[0] in 'AaTt':
            roo2[0] = np.exp(-comp.dg[0]-dgt0-dgt)

        if comp.seq[-1] in 'AaTt':
            roo2[-2] = np.exp(-comp.dg[-1]+dgt0-dgt)
        return roo2

    def __transitions(self, nbases, inds, comp) -> np.ndarray:
        roo  = self.__roo(comp)
        roo2 = self.__roo2(comp)
        mat  = np.zeros((inds.max()+1,)*2, dtype = 'f8')
        rco  = self.elasticity.rco(self.temperatures)
        for in0, in1, j, dist in chain(
                self.__iter5prime(inds),
                self.__iter3prime(inds)
        ):
            mat[in0, in1] += rco # closing
            mat[in1, in1] += -rco # closing
            mat[in1, in0] +=  (roo if dist else roo2)[j] # opening
            mat[in0, in0] += -(roo if dist else roo2)[j] # opening


        #escaping terms
        for in0, i in self.__iterescaping(nbases, inds):
            mat[in0,in0] += -roo2[i]
        return mat

    def __encircling(self, cordsbp, ind, mat):
        r"""
         \
          \___/
         ------
        5'->i = 2  3'->j=5
        5' end transitions possible state
        """
        # encircling transitions
        # encircling rate for one bounded bp of the oligo
        nbases = ind.shape[0]
        ren    = self.salt.encirclingrate(self, cordsbp)
        for i in range(3):
            in0           = ind[i,nbases-i-1]
            mat[in0,in0] += -ren**(nbases-i)

    def __fork(self, comp, ind, mat):
        """ 5' end transitions possible state"""
        # hairpin transitions possible state
        nbases = ind.shape[0]
        rch    = self.elasticity.rch(self.temperatures)
        roh    = np.append(np.exp(-comp.dgh),    0)
        for i in range(1, nbases):
            for j in range(i):
                for k in range(i+1,nbases):
                    in0 = ind[i,j,k]
                    in1 = ind[i,j+1,k]
                    mat[in0,in1] += roh[j]  # opening
                    mat[in1,in1] += -roh[j] # opening
                    mat[in1,in0] += rch     # closing
                    mat[in0,in0] += -rch    # closing

    def __zipping(self, seq, dg, delta):
        key = KeyComputer(seq[0], seq[1]).key
        tab = self.table
        for i in range(len(seq[0]) - 1):
            tmp = tab.get(key(i), None)
            if tmp is None:
                # We haven't found the key...
                raise KeyError(f"Base not found {i}, {seq[0][i:i+2]}/{seq[1][i:i+2]}")

            if delta is not None:
                delta += tmp
            dg[i] += self.dgcor(tmp)
