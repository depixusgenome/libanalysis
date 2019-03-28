#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"Computing melting times"
from typing     import Tuple, cast
import numpy  as np
try:
    # pylint: disable=unused-import
    from sequences.translator import complement, gccontent
except ImportError:
    # stay BioPython compatible
    from Bio.SeqUtils import GC as gccontent # type: ignore
    from Bio.Seq      import Seq
    def complement(seq:str) -> str:
        "return the complement"
        return str(Seq(seq).complement())

from   ._data       import nndata, R, T0, NNDATA

SaltInfo = Tuple[float, float, float, float]
class Salt: # pylint: disable=too-many-instance-attributes
    "salt"
    def __init__(self, **_):
        self.method          = 5
        self.encercling      = False
        self.encirclingcoeff = 0.34
        self.encirclingbp    = 2
        self.dsCharge = 0.4
        self.rhoNa    = 150
        self.rhoK     = 0
        self.rhoTris  = 30
        self.rhoMg    = 0
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

    def compute( # pylint: disable=too-many-arguments
            self, seq, rhoseq, rhooligo, mtg, delta
    ) -> SaltInfo:
        "compute salt corrections"
        rlogk            = R*np.log((rhoseq - (rhooligo / 2.0)) * 1e-9)
        delta_h, delta_s = delta*[1e3, 1]
        corrh            = self.salt_correction(
            Na     = self.rhoNa,
            K      = self.rhoK,
            Tris   = self.rhoTris,
            Mg     = self.rhoMg,
            dNTPs  = self.rhodNTPs,
            method = self.method,
            seq    = seq
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
        return (
            (cords/(len(seq)-1))* mtg,
            (delta_sc - delta_s)/(len(seq)-1),
            meltingtemp,
            delta_h*1e-3 - (delta_sc) *mtg
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
    def __init__(self, **_):
        self.dG = 25
        for i in set(self.__dict__) & set(_):
            setattr(self, i, _[i])
    tG  = cast(float, property(lambda self: self.dG+T0))
    mtG = cast(float, property(lambda self: (self.dG+T0)*1e-3))
    ft  = cast(float, property(lambda self: 4.1*self.tG/(T0+25)))

class SingleStrandModel:
    "elastic model coefficients for single-strand dna"
    def __init__(self, **_):
        self.force = 8.5
        self.bpss  = 2.14
        self.dss   = 0.542
        self.sss   = 216
        for i in set(self.__dict__) & set(_):
            setattr(self, i, _[i])

    def gds(self, temperatures):
        """ elasticity models for the closing rates for the ds DNA"""
        ft    = temperatures.ft
        force = self.force
        return (force/ft - np.sqrt(force/ft/50) + 0.5*force**2./ft/1230)*0.34

    def gss(self, temperatures):
        """ elasticity models for the closing rates for the ss DNA"""
        force = self.force
        ft    = temperatures.ft
        u     = self.bpss*force/ft
        return (
            0 if u == 0 else
            self.dss/self.bpss*(np.log(np.sinh(u)/u) + 0.5*force**2./ft/self.sss)
        )

    def rco(self, temperatures):
        """closing rates:only depend on the force"""
        return np.exp(-self.gss(temperatures)+self.gds(temperatures))

    def rch(self, temperatures):
        """closing rates:only depend on the force"""
        return np.exp(-2*self.gss(temperatures))

class BaseComputations: # pylint: disable=too-many-instance-attributes
    """
    base data
    """
    def __init__(self, **_):
        self.table        : NNDATA            = {}
        self.settables()
        self.salt         : Salt              = Salt()
        self.temperatures : Temperatures      = Temperatures(**_)
        self.elasticity   : SingleStrandModel = SingleStrandModel(**_)
        for i in set(self.__dict__) & set(_):
            setattr(self, i, _[i])

    force = cast(
        float,
        property(
            lambda self: self.elasticity.force,
            lambda self, val: setattr(self.elasticity, 'force', val)
        )
    )

    def settables(
            self,
            nomiss       = ("DNA_NN3", "LNA_DNA_NN2"),
            internalmiss = ("DNA_IMM1", "LNA_DNA_IMM1"),
            dangling     = "DNA_DE1",
            terminal     = ("DNA_TMM1", "LNA_DNA_TMM1")
    ):
        """
        sets the table for non-missmatches (1), internal missmatches (2) and
        dangling ends (3)
        """
        self.table = nndata(nomiss, internalmiss, dangling, terminal)
        return self

    def setmode(self, mode:str):
        "sets the computation mode"
        if mode == 'oligo':
            self.salt.encercling  = False
            self.elasticity.force = 0
        elif mode == 'fork':
            self.salt.encercling  = False
        elif mode == 'apex':
            self.salt.encercling  = True
            self.elasticity.force = 5.
        return self

    gds = property(lambda self: self.elasticity.gds(self.temperatures))
    gss = property(lambda self: self.elasticity.gss(self.temperatures))
    cor = property(lambda self: 1.0/(R*self.temperatures.mtG)) # cal/conc_K-mol

    def dgcor(self, delta: np.ndarray) -> float:
        "corrected enthalpy"
        return -self.cor*(delta[0]-delta[1]*self.temperatures.mtG)
