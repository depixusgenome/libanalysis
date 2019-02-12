#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"Computing melting times"
from itertools  import chain
from typing     import Tuple, Iterator, Callable, cast
import re
import numpy as np
from   ..translator import complement, gccontent
from   ._data       import nndata, R, T0, NNDATA

class Salt: # pylint: disable=too-many-instance-attributes
    "salt"
    def __init__(self, **_):
        self.encercling      = False
        self.encirclingcoeff = 0.34
        self.encirclingbp    = 2
        self.dsCharge = 0.4
        self.rhoNa    = 50
        self.rhoK     = 0
        self.rhoTris  = 0
        self.rhoMg    =0
        self.rhodNTPs = 0
        for i in set(self.__dict__) & set(_):
            setattr(self, i, _[i])

    def encirclingrate(self, cnf,  cordsbp)-> float:
        "return encircling rate"
        if self.encercling:
            tmpdg = -2*cnf.gss -cnf.gds
            # must depend on salt corr
            tmpdg += self.encirclingbp * cordsbp * cnf.temperatures.mtG
            return self.encirclingcoeff*np.exp(tmpdg)  # was 0.1
        return 0.

    @property
    def method(self):
        "salt correction config"
        return self.__dict__['method']

    @method.setter
    def method(self, val):
        "salt correction config"
        self.__dict__['method'] = val
        if val == 6:
            self.dsCharge        = 0.5
            self.encirclingbp    = 2
            self.encirclingcoeff = 0.27
        elif val == 5:
            self.dsCharge        = 0.4
            self.encirclingbp    = 2
            self.encirclingcoeff = 0.3

    def compute(self, rhoseq, rhooligo, mtg, comp):
        "compute salt corrections"
        rlogk            = R*np.log((rhoseq - (rhooligo / 2.0)) * 1e-9)
        delta_h, delta_s = comp.delta*[1e3, 1]
        corrh            = self.salt_correction(
            Na     = self.rhoNa,
            K      = self.rhoK,
            Tris   = self.rhoTris,
            Mg     = self.rhoMg,
            dNTPs  = self.rhodNTPs,
            method = self.method,
            seq    = comp.oseq
        )
        corr  =  self.dsCharge * corrh

        # we compute salt correction for normal oligo to compute Kd and encercling energy
        if self.method in range(1,4):
            meltingtemp  = delta_h / (delta_s + rlogk) - T0 + corr
            delta_sc     = delta_h / (meltingtemp + T0) - rlogk
            cords        = self.dsCharge * (delta_sc - delta_s)
        elif self.method == 5:
            cords       = corr
            meltingtemp = delta_h / (delta_s + corrh + rlogk) - T0
            delta_sc    = delta_s + corrh
        else:
            meltingtemp = delta_h / (delta_s + rlogk) - T0
            meltingtemp = 1 / (1 / (meltingtemp + T0) + corrh) - T0
            delta_sc    = delta_h / (meltingtemp + T0) - rlogk
            cords       = self.dsCharge * (delta_sc - delta_s)

            meltingtemp = delta_h / (delta_s + rlogk) - T0
            meltingtemp = 1 / (1 / (meltingtemp + T0) + corr) - T0
            delta_sc    = delta_h / (meltingtemp + T0) - rlogk

        # energy between the oligo and template (can be LNA/DNA)
        delta     = cords*(cords/(len(comp.oseq)-1))* mtg
        comp.dg  += delta
        comp.dgh += delta
        return (
            meltingtemp,
            delta_h - (delta_sc) *mtg,
            (delta_sc - delta_s)/(len(comp.oseq)-1)
        )

    # Extracted as is from Bio.SeqUtils.MeltingTemp
    # pylint: disable=invalid-name,too-many-arguments,too-many-locals,unneeded-not
    # pylint: disable=no-else-return,too-many-branches,bad-continuation
    @staticmethod
    def salt_correction(Na=0, K=0, Tris=0, Mg=0, dNTPs=0, method=1, seq=None):
        """
        Calculate a term to correct Tm for salt ions.

        Depending on the Tm calculation, the term will correct Tm or entropy. To
        calculate corrected Tm values, different operations need to be applied:

         - methods 1-4: Tm(new) = Tm(old) + corr
         - method 5: deltaS(new) = deltaS(old) + corr
         - methods 6+7: Tm(new) = 1/(1/Tm(old) + corr)

        Parameters:
         - Na, K, Tris, Mg, dNTPS: Millimolar concentration of respective ion. To
           have a simple 'salt correction', just pass Na. If any of K, Tris, Mg and
           dNTPS is non-zero, a 'sodium-equivalent' concentration is calculated
           according to von Ahsen et al. (2001, Clin Chem 47: 1956-1961):
           [Na_eq] = [Na+] + [K+] + [Tris]/2 + 120*([Mg2+] - [dNTPs])^0.5
           If [dNTPs] >= [Mg2+]: [Na_eq] = [Na+] + [K+] + [Tris]/2
         - method: Which method to be applied. Methods 1-4 correct Tm, method 5
           corrects deltaS, methods 6 and 7 correct 1/Tm. The methods are:

           1. 16.6 x log[Na+]
              (Schildkraut & Lifson (1965), Biopolymers 3: 195-208)
           2. 16.6 x log([Na+]/(1.0 + 0.7*[Na+]))
              (Wetmur (1991), Crit Rev Biochem Mol Biol 126: 227-259)
           3. 12.5 x log(Na+]
              (SantaLucia et al. (1996), Biochemistry 35: 3555-3562
           4. 11.7 x log[Na+]
              (SantaLucia (1998), Proc Natl Acad Sci USA 95: 1460-1465
           5. Correction for deltaS: 0.368 x (N-1) x ln[Na+]
              (SantaLucia (1998), Proc Natl Acad Sci USA 95: 1460-1465)
           6. (4.29(%GC)-3.95)x1e-5 x ln[Na+] + 9.40e-6 x ln[Na+]^2
              (Owczarzy et al. (2004), Biochemistry 43: 3537-3554)
           7. Complex formula with decision tree and 7 empirical constants.
              Mg2+ is corrected for dNTPs binding (if present)
              (Owczarzy et al. (2008), Biochemistry 47: 5336-5353)

        Examples
        --------
        >>> from Bio.SeqUtils import MeltingTemp as mt
        >>> print('%0.2f' % mt.salt_correction(Na=50, method=1))
        -21.60
        >>> print('%0.2f' % mt.salt_correction(Na=50, method=2))
        -21.85
        >>> print('%0.2f' % mt.salt_correction(Na=100, Tris=20, method=2))
        -16.45
        >>> print('%0.2f' % mt.salt_correction(Na=100, Tris=20, Mg=1.5, method=2))
        -10.99

        """
        # Extracted as is from Bio.SeqUtils.MeltingTemp
        if method in (5, 6, 7) and not seq:
            raise ValueError('sequence is missing (is needed to calculate ' +
                             'GC content or sequence length).')
        if seq:
            seq = str(seq)
        corr = 0
        if not method:
            return corr
        Mon = Na + K + Tris / 2.0  # Note: all these values are millimolar
        mg = Mg * 1e-3             # Lowercase ions (mg, mon, dntps) are molar
        # Na equivalent according to von Ahsen et al. (2001):
        if sum((K, Mg, Tris, dNTPs)) > 0 and not method == 7 and dNTPs < Mg:
            # dNTPs bind Mg2+ strongly. If [dNTPs] is larger or equal than
            # [Mg2+], free Mg2+ is considered not to be relevant.
            Mon += 120 * np.sqrt(Mg - dNTPs)
        mon = Mon * 1e-3
        # Note: np.log = ln(), np.log10 = log()
        if method in range(1, 7) and not mon:
            raise ValueError('Total ion concentration of zero is not allowed in ' +
                             'this method.')
        if method == 1:
            corr = 16.6 * np.log10(mon)
        if method == 2:
            corr = 16.6 * np.log10((mon) / (1.0 + 0.7 * (mon)))
        if method == 3:
            corr = 12.5 * np.log10(mon)
        if method == 4:
            corr = 11.7 * np.log10(mon)
        if method == 5:
            corr = 0.368 * (len(seq) - 1) * np.log(mon)
        if method == 6:
            corr = (4.29 * gccontent(seq) / 100 - 3.95) * 1e-5 * np.log(mon) +\
                9.40e-6 * np.log(mon) ** 2
        if method == 7:
            a, b, c, d = 3.92, -0.911, 6.26, 1.42
            e, f, g = -48.2, 52.5, 8.31
            if dNTPs > 0:
                dntps = dNTPs * 1e-3
                ka = 3e4  # Dissociation constant for Mg:dNTP
                # Free Mg2+ calculation:
                mg = (-(ka * dntps - ka * mg + 1.0) +
                      np.sqrt((ka * dntps - ka * mg + 1.0) ** 2 +
                                4.0 * ka * mg)) / (2.0 * ka)
            if Mon > 0:
                r = np.sqrt(mg) / mon
                if r < 0.22:
                    corr = (4.29 * gccontent(seq) / 100 - 3.95) * \
                        1e-5 * np.log(mon) + 9.40e-6 * np.log(mon) ** 2
                    return corr
                elif r < 6.0:
                    a = 3.92 * (0.843 - 0.352 * np.sqrt(mon) * np.log(mon))
                    d = 1.42 * (1.279 - 4.03e-3 * np.log(mon) -
                                8.03e-3 * np.log(mon) ** 2)
                    g = 8.31 * (0.486 - 0.258 * np.log(mon) +
                                5.25e-3 * np.log(mon) ** 3)
            corr = (a + b * np.log(mg) + (gccontent(seq) / 100) *
                    (c + d * np.log(mg)) + (1 / (2.0 * (len(seq) - 1))) *
                    (e + f * np.log(mg) + g * np.log(mg) ** 2)) * 1e-5
        if method > 7:
            raise ValueError('Allowed values for parameter \'method\' are 1-7.')
        return corr

class Temperatures:
    "temperatures"
    def __init__(self):
        self.dG = 25
    tG  = cast(float, property(lambda self: self.dG+273.5))
    mtG = cast(float, property(lambda self: (self.dG+273.5)*1e-3))
    ft  = cast(float, property(lambda self: 4.1*self.tG/(T0+25)))

class SingleStrandModel:
    "elastic model coefficients for single-strand dna"
    def __init__(self):
        self.bpss = 2.14
        self.dss  = 0.542
        self.sss  = 216

    @staticmethod
    def gds(force, temperatures):
        """ elasticity models for the closing rates for the ds DNA"""
        ft  = temperatures.ft
        return (force/ft - np.sqrt(force/ft/50) + 0.5*force**2./ft/1230)*0.34

    def gss(self, force, temperatures):
        """ elasticity models for the closing rates for the ss DNA"""
        ft = temperatures.ft
        u  = self.bpss*force/ft
        return (
            0 if u == 0 else
            self.dss/self.bpss*(np.log(np.sinh(u)/u) + 0.5*force**2./ft/self.sss)
        )

    def rco(self, force, temperatures):
        """closing rates:only depend on the force"""
        return np.exp(-self.gss(force, temperatures)+self.gds(force, temperatures))

    def rch(self, force, temperatures):
        """closing rates:only depend on the force"""
        return np.exp(-2*self.gss(force, temperatures))

class KeyComputer:
    "computes keys for various energy tables"
    def __init__(self, seq, oligo):
        self.seq   = str(seq)
        self.oligo = str(oligo if oligo else complement(seq))

    def key(self, ind, order = True, seq = True) -> str:
        "get table keys"
        seq, cseq = (self.seq, self.oligo)[::1 if seq else -1]
        iord      = 1 if order else -1
        if ind > 0:
            return seq[:2*ind][::iord] + '/' + cseq[:2*ind][::iord]
        return seq[2*ind:][::iord] + '/' + cseq[2*ind:][::iord]

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
    def __init__(self, seq, oligo, oligotype):
        super().__init__(seq, oligo)
        self.oseq   = self.seq
        self.ooligo = self.oligo
        self.otype  = oligotype
        if oligotype in {'LNA', 'LNA2'}:
            self.seq    = self.oseq.lower()
            self.oligo  = self.ooligo.lower()
        elif oligotype in {'DNA'}:
            self.seq    = self.oseq.upper()
            self.oligo  = self.ooligo.upper()

        self.delta = np.zeros(2, dtype = 'f8')

    def resetdg(self):
        "sets dg & dgh"
        self.dg    = np.zeros(len(self.oseq)-1, dtype = 'f8')
        self.dgh   = np.zeros(len(self.oseq)-1, dtype = 'f8')

    def hairpinseq(self) -> Tuple[str, str]:
        "return the hairpin sequence"
        keys = self.oseq, self.ooligo
        return (
            (keys[0].lower(), keys[1].lower()) if self.otype in {'LNA', 'LNA2'} else
            (keys[0].upper(), keys[1].upper()) if self.otype in {'DNA'}         else
            keys
        )

class StateMatrixComputer: # pylint: disable=too-many-instance-attributes
    """
    Return the Tm using nearest neighbor thermodynamics.

    Arguments:
     - sequence: The primer/probe sequence as string or Biopython sequence object.
       For RNA/DNA hybridizations sequence must be the RNA sequence.
     - oligo: Complementary sequence. The sequence of the template/target in
       3'->5' direction. oligo is necessary for mismatch correction and
       dangling-ends correction. Both corrections will automatically be
       applied if mismatches or dangling ends are present. Default=None.
     - shift: Shift of the primer/probe sequence on the template/target
       sequence, e.g.::

                           shift=0       shift=1        shift= -1
        Primer (sequence): 5' ATGC...    5'  ATGC...    5' ATGC...
        Template (oligo):  3' TACG...    3' CTACG...    3'  ACG...

       The shift parameter is necessary to align sequence and oligo if they have
       different lengths or if they should have dangling ends. Default=0
     - table: Thermodynamic NN values, eight tables are implemented:
       For DNA/DNA hybridizations:

        - DNA_NN1: values from Breslauer et al. (1986)
        - DNA_NN2: values from Sugimoto et al. (1996)
        - DNA_NN3: values from Allawi & SantaLucia (1997) (default)
        - DNA_NN4: values from SantaLucia & Hicks (2004)

       For RNA/RNA hybridizations:

        - RNA_NN1: values from Freier et al. (1986)
        - RNA_NN2: values from Xia et al. (1998)
        - RNA_NN3: valuse from Chen et al. (2012)

       For RNA/DNA hybridizations:

        - R_DNA_NN1: values from Sugimoto et al. (1995)

       Use the module's maketable method to make a new table or to update one
       one of the implemented tables.
     - self.tmm: Thermodynamic values for terminal mismatches.
       Default: DNA_TMM1 (SantaLucia & Peyret, 2001)
     - self.imm: Thermodynamic values for internal mismatches, may include
       insosine mismatches. Default: DNA_IMM1 (Allawi & SantaLucia, 1997-1998
       Peyret et al., 1999; Watkins & SantaLucia, 2005)
     - self.dangling: Thermodynamic values for dangling ends:

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
        self.force        : float             = 8.5
        self.rate         : float             = (4/1.4)*1e-6
        self.fork_tau_mul : float             = 1.
        self.nn           : NNDATA            = nndata("DNA_NN3")
        self.tmm          : NNDATA            = nndata("DNA_TMM1")
        self.imm          : NNDATA            = nndata("DNA_IMM1", "DNA_NN3")
        self.dangling     : NNDATA            = nndata("DNA_DE1")
        self.salt         : Salt              = Salt()
        self.loop         : bool              = False
        self.temperatures : Temperatures      = Temperatures()
        self.elasticity   : SingleStrandModel = SingleStrandModel()

    def setmode(self, mode:str):
        "sets the computation mode"
        if mode == 'oligo':
            self.salt.encercling = False
            self.loop            = True
            self.force           = 0
        elif mode == 'fork':
            self.loop            = False
            self.force           = 8.5
            self.fork_tau_mul    = 2.8
        elif mode == 'apex':
            self.salt.encercling = True
            self.force           = 5.
            self.loop            = True
        return self

    def states(
            self,
            comp,
            shift    : int = 0,
            rhoseq   : int = 25,
            rhooligo : int = 25
    ):
        "return the states matrix"
        self.__shift(shift, comp) # compute resized sequences, withouth overdangling ends
        comp.resetdg()            # make sure dg & dgh have updated sizes

        # Now for terminal mismatches
        key    = comp.key(1, False, False)
        off_bp = key in self.tmm
        if off_bp:
            comp.delta += self.tmm[key]
            comp.dg[0] += self.__dgcor(self.tmm[key])
            comp.pop(0)

        key = comp.key(-1, True, comp.otype in {'LNA', 'LNA2'})
        if key in self.tmm:
            comp.delta  += self.tmm[key]
            comp.dg[-1] += -self.__dgcor(self.tmm[key])
            comp.pop(-1)

        # Now everything 'unusual' at the ends is handled and removed and we can
        # look at the initiation.
        # One or several of the following initiation types may apply:

        # Type: General initiation value
        comp.delta += self.nn['init']

        # Type: Duplex with no (allA/T) or at least one (oneG/C) GC pair
        comp.delta += self.nn['init_'+('oneG/C' if gccontent(comp.oseq.upper()) else 'allA/T')]

        # Type: Penalty if 5' end is T
        comp.delta += self.nn['init_5T/A'] * ((comp.oseq[0] in 'Tt')+(comp.oseq[-1] in 'Aa'))

        # Type: Different values for G/C or A/T terminal basepairs
        ends = (comp.oseq[0] + comp.oseq[-1]).upper()
        for tpe in ("AT", "GC"):
            tmp         = self.nn[f'init_{tpe[0]}/{tpe[1]}']
            comp.delta += sum(ends.count(i) for i in tpe) * np.array(tmp)

        dgt0         = self.__dgcor(self.nn['init_A/T'] - self.nn['init_G/C'])
        comp.dg[0]  += dgt0*(comp.oseq[0]  in 'AaTt')
        comp.dg[-1] += dgt0*(comp.oseq[-1] in 'AaTt')

        # Finally, the 'zipping'
        self.__zipping((comp.seq, comp.oligo), comp.dg[off_bp:], comp.delta)
        self.__zipping(comp.hairpinseq,        comp.dgh, None)

        # We compute salt correction for the hybridized oligo that is with
        # reduced charge near dsDNA

        # roo[0] opening of the first 5' base of the oligo towards the fork
        # l is an index over all possible configurations
        #ind[i, j]  is the index of the configuration :
        # i = position of the opening front on the 5' end, [0, nn-1]
        # j = position of the opening front on the 3' end, nn-1 nothing open
        # loop version
        nbases, inds = self.__state_indexes(comp)
        mat          = self.__transitions(nbases, inds, comp)
        if self.loop:
            self.__encircling((rhoseq, rhooligo), comp, inds, mat)
        else:
            self.__fork(comp, inds, mat)
        return mat, inds

    def __call__( # pylint: disable=too-many-arguments
            self,
            sequence : str,
            oligo    : str,
            shift    : int = 0,
            oligotype: str = 'DNA',
            rhoseq   : int = 25,
            rhooligo : int = 25
    ):
        comp        = ComputationDetails(sequence, oligo, oligotype)
        temp, delta = self.salt.compute(rhoseq, rhooligo, self.temperatures.mtG, comp)[:-1]

        trep = self.__trep(comp, *self.states(comp, shift, rhoseq, rhooligo))
        dgf  = self.cor*delta+((len(comp.oseq)-1)*(self.gss-self.gds))
        return (
            trep,
            1/trep,                         # kon
            1/(trep*np.exp(dgf) * 1000000), # koff
            dgf,
            temp
        )

    gds = property(lambda self: self.elasticity.gds(self.force, self.temperatures))
    gss = property(lambda self: self.elasticity.gss(self.force, self.temperatures))
    cor = property(lambda self: 1.0/(R*self.temperatures.mtG)) # cal/conc_K-mol

    def __trep(self, comp, mat, inds):
        #initial state: nn open bases for the hairpin 0 open bases for the oligo
        ini                                                    = np.zeros(mat.shape[0])
        ini[inds[(0,)*(len(mat.shape)-1)+(len(comp.oseq)-1,)]] = 1

        #holding state: all possible configurations
        hold                                                   = np.ones(mat.shape[0])

        # between the initil and final state
        trep =-self.rate*(hold @ (np.matrix(mat).I @ ini))[0,0]
        if not self.loop:
            trep *= self.fork_tau_mul
        return trep

    def __dgcor(self, delta):
        return -self.cor*(delta[0]-delta[1]*self.temperatures.mtG)

    @staticmethod
    def __delta(table, comp, *args) -> np.ndarray:
        key = comp.key(*args)
        return table[key] if key in table else np.zeros(2, dtype = 'f8')

    __RESEQ = re.compile(r"^\.*?(?P<SEQ>\.?[atgcATGC]*\.?)\.*$")
    def __shift(self, shift:int, comp:ComputationDetails):
        # Dangling ends?
        if shift or len(comp.seq) != len(comp.oligo):
            # Align both sequences using the shift parameter
            comp.seq    = '.' * max(0,  shift) + comp.seq
            comp.oligo  = '.' * max(0, -shift) + comp.oligo
            comp.seq   += '.' * max(0, len(comp.oligo) - len(comp.seq))
            comp.oligo += '.' * max(0, len(comp.seq) -  len(comp.oligo))

            # Remove 'over-dangling' ends
            comp.seq    = self.__RESEQ.match(comp.seq).group("SEQ")  # type: ignore
            comp.oligo  = self.__RESEQ.match(comp.oligo).group("SEQ") # type: ignore

            # Now for the dangling ends
            for ind in (0, -1):
                if any(i[ind] == "." for i in (comp.seq, comp.oligo)):
                    comp.delta += self.__delta(self.dangling, comp, ind, ind == 0, ind == 0)
                    comp.pop(ind)
            comp.oseq   = comp.seq
            comp.ooligo = comp.oligo

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
    def __iter5prime(nbases, inds):
        if len(inds.shape) == 2:
            yield from (
                (inds[i,j], inds[i+1,j], i, j-i > 2)
                for j in range(2,nbases) for i in range(j-1)
            )

        yield from (
            (inds[i,j,k], inds[i+1,j, k], i, k-i > 2)
            for j in range (0,nbases-1) for i in range(j,nbases-2) for k in range (i+2,nbases)
        )

    @staticmethod
    def __iter3prime(nbases, inds):
        if len(inds.shape) == 2:
            yield from (
                (inds[i,j], inds[i,j-1], j-1, j-i > 2)
                for i in range(nbases) for j in range(i+2, nbases)
            )
        yield from (
            (inds[i,j,k], inds[i,j,k-1], k-1, k-i > 2)
            for i in range(nbases) for j in range(i+1, nbases) for k in range(i+2, nbases)
        )

    @staticmethod
    def __iterescaping(nbases, inds):
        if len(inds.shape) == 2:
            yield from ((inds[i,i+1], i) for i in range(nbases-1))

        yield from ((inds[i,j,i+1], i) for i in range (nbases-1) for j in range(i+1))

    def __roo(self, comp):
        dgt0 = self.__dgcor(self.nn['init_A/T'] - self.nn['init_G/C'])
        roo  = np.append(np.exp(-comp.dg),     0)
        if comp.seq[0] in 'AaTt':
            roo[0]  = np.exp(-comp.dg[0]-dgt0)
        if comp.seq[-1] in 'AaTt':
            roo[-2] = np.exp(-comp.dg[-1]+dgt0)
        return roo

    def __roo2(self, comp):
        dgt0 = self.__dgcor(self.nn['init_A/T'] - self.nn['init_G/C'])
        dgt  = self.__dgcor(self.nn['init_G/C'])
        roo2 = np.append(np.exp(-comp.dg-dgt), 0)
        if comp.seq[0] in 'AaTt':
            roo2[0] = np.exp(-comp.dg[0]-dgt0-dgt)

        if comp.seq[-1] in 'AaTt':
            roo2[-2] = np.exp(-comp.dg[-1]+dgt0-dgt)
        return roo2

    def __transitions(self, nbases, inds, comp) -> np.ndarray:
        roo  = self.__roo(comp)
        roo2 = self.__roo2(comp)
        mat  = np.zeros((inds[-1],)*len(inds.shape))
        rco  = self.elasticity.rco(self.force, self.temperatures)
        for in0, in1, j, dist in chain(
                self.__iter5prime(nbases, inds),
                self.__iter3prime(nbases, inds)
        ):
            mat[in0, in1] += rco # closing
            mat[in1, in1] += -rco # closing
            mat[in1, in0] +=  (roo if dist else roo2)[j] # opening
            mat[in0, in0] += -(roo if dist else roo2)[j] # opening

        #escaping terms
        for in0, i in self.__iterescaping(nbases, inds):
            mat[in0,in0] += -roo2[i]
        return mat

    def __encircling(self, rhos, comp, ind, mat):
        r"""
         \
          \___/
         ------
        5'->i = 2  3'->j=5
        5' end transitions possible state
        """
        cordsbp = self.salt.compute(*rhos, self.temperatures.mtG, comp)[-1]
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
        rch    = self.elasticity.rch(self.force, self.temperatures)
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
        tab = self.imm
        for i in range(len(seq[0]) - 1):
            neighbors = (key(i, k, True) for k in (True, False))
            tmp       = next((tab[j] for j in neighbors if j in tab), None)
            if tmp is None:
                # We haven't found the key...
                raise KeyError(f"Base not found $i")

            if delta is not None:
                delta += tmp
            dg[i] += -self.__dgcor(tmp)
