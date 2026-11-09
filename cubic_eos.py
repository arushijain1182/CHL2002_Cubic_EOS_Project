#!/usr/bin/env python3
"""
cubic_eos.py - PVT behaviour of real gases/liquids from cubic equations of state
================================================================================

Implements the generic cubic equation of state (Smith, Van Ness, Abbott &
Swihart, Ch. 3, Eqs. 3.41-3.51):

        P = RT/(V - b) - a(T) / ((V + eps*b)(V + sigma*b))

Equivalent form used in lectures (u = sigma + eps, w = sigma*eps):

        P = RT/(V - b) - a(T) / (V^2 + u*b*V + w*b^2)

                    u    w     a(T)
        vdW         0    0     constant
        RK          1    0     a_c * Tr^-1/2             (function of T)
        SRK         1    0     a_c * alpha(Tr, omega)    (function of T, omega)
        PR          2   -1     a_c * alpha(Tr, omega)    (function of T, omega)

(Note: "omega" in the a(T) column is the acentric factor; the w above is not.)
with parameter assignments from Table 3.1 for

        vdW  (van der Waals, 1873)
        RK   (Redlich/Kwong, 1949)
        SRK  (Soave/Redlich/Kwong, 1972)
        PR   (Peng/Robinson, 1976)

Capabilities
------------
  * Z, V, density at given (T, P)         -> all real roots + stable-phase pick
  * P at given (T, V)                     -> direct
  * T at given (P, V)                     -> 1-D root find
  * Saturation pressure Psat(T) and saturated liquid/vapour volumes
    (equal-fugacity criterion)
  * Fugacity coefficient
  * Isotherm tables and PV / Z-P plots
  * Comparison against ideal gas and the Pitzer second-virial correlation
  * Residual and total enthalpy / entropy (H^R, S^R, H, S), property changes between
    any two states (including across a phase change), latent heat of vaporization
  * Species data for 89 compounds taken directly from the book (Tables B.1 and C.1)
  * Command-line interface  (python cubic_eos.py --help)

Units (internal): T [K], P [bar], V [cm^3/mol], R = 83.14 cm^3 bar /(mol K)
Dependencies: numpy (matplotlib only for plots).
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

R_GAS = 83.14462618  # cm^3 bar / (mol K)


# ----------------------------------------------------------------------------
# Equation-of-state definitions  (Table 3.1)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EOS:
    name: str
    sigma: float
    eps: float
    Omega: float
    Psi: float
    alpha: Callable[[float, float], float]  # alpha(Tr, omega)
    Zc: float                               # EOS-implied critical Z (info only)

    @property
    def u(self) -> float:
        """Lecture notation: V^2 + u*b*V + w*b^2 in the denominator, u = sigma + eps."""
        return self.sigma + self.eps

    @property
    def w(self) -> float:
        """Lecture notation: w = sigma * eps."""
        return self.sigma * self.eps


def _alpha_vdw(Tr: float, w: float) -> float:
    return 1.0


def _alpha_rk(Tr: float, w: float) -> float:
    return Tr ** -0.5


def _alpha_srk(Tr: float, w: float) -> float:
    return (1.0 + (0.480 + 1.574 * w - 0.176 * w * w) * (1.0 - math.sqrt(Tr))) ** 2


def _alpha_pr(Tr: float, w: float) -> float:
    return (1.0 + (0.37464 + 1.54226 * w - 0.26992 * w * w) * (1.0 - math.sqrt(Tr))) ** 2


EOS_MODELS: Dict[str, EOS] = {
    "vdW": EOS("vdW", 0.0, 0.0, 1 / 8, 27 / 64, _alpha_vdw, 3 / 8),
    "RK": EOS("RK", 1.0, 0.0, 0.08664, 0.42748, _alpha_rk, 1 / 3),
    "SRK": EOS("SRK", 1.0, 0.0, 0.08664, 0.42748, _alpha_srk, 1 / 3),
    "PR": EOS("PR", 1 + math.sqrt(2), 1 - math.sqrt(2), 0.07780, 0.45724, _alpha_pr, 0.30740),
}


# ----------------------------------------------------------------------------
# Species data (critical constants, acentric factor, molar mass, ideal-gas Cp)
# ----------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Species data, copied from the textbook (Smith, Van Ness, Abbott, Swihart,
# "Introduction to Chemical Engineering Thermodynamics", 8th ed.)
#   _B1  : Table B.1  -> M [g/mol], omega, Tc [K], Pc [bar], Zc, Vc [cm3/mol], Tn [K]
#   _C1  : Table C.1  -> ideal-gas Cp/R = A + B*T + C*T^2 + D/T^2 (T in K);
#                        tuple = (A, B, C, D, Tmax)
# ---------------------------------------------------------------------------
_B1 = {
    '1,3-butadiene': {'M': 54.092, 'Pc': 42.77, 'Tc': 425.2, 'Tn': 268.7, 'Vc': 220.4, 'Zc': 0.267, 'omega': 0.19},
    '1-butanol': {'M': 74.123, 'Pc': 44.23, 'Tc': 563.1, 'Tn': 390.8, 'Vc': 275.0, 'Zc': 0.26, 'omega': 0.594},
    '1-butene': {'M': 56.108, 'Pc': 40.43, 'Tc': 420.0, 'Tn': 266.9, 'Vc': 239.3, 'Zc': 0.277, 'omega': 0.191},
    '1-hexanol': {'M': 102.177, 'Pc': 35.1, 'Tc': 611.4, 'Tn': 430.6, 'Vc': 381.0, 'Zc': 0.263, 'omega': 0.579},
    '1-hexene': {'M': 84.161, 'Pc': 31.4, 'Tc': 504.0, 'Tn': 336.3, 'Vc': 354.0, 'Zc': 0.265, 'omega': 0.28},
    '1-propanol': {'M': 60.096, 'Pc': 51.75, 'Tc': 536.8, 'Tn': 370.4, 'Vc': 219.0, 'Zc': 0.254, 'omega': 0.622},
    '2-propanol': {'M': 60.096, 'Pc': 47.62, 'Tc': 508.3, 'Tn': 355.4, 'Vc': 220.0, 'Zc': 0.248, 'omega': 0.668},
    'acetaldehyde': {'M': 44.053, 'Pc': 55.5, 'Tc': 466.0, 'Tn': 294.0, 'Vc': 154.0, 'Zc': 0.221, 'omega': 0.291},
    'acetic acid': {'M': 60.053, 'Pc': 57.86, 'Tc': 592.0, 'Tn': 391.1, 'Vc': 179.7, 'Zc': 0.211, 'omega': 0.467},
    'acetone': {'M': 58.08, 'Pc': 47.01, 'Tc': 508.2, 'Tn': 329.4, 'Vc': 209.0, 'Zc': 0.233, 'omega': 0.307},
    'acetonitrile': {'M': 41.053, 'Pc': 48.3, 'Tc': 545.5, 'Tn': 354.8, 'Vc': 173.0, 'Zc': 0.184, 'omega': 0.338},
    'acetylene': {'M': 26.038, 'Pc': 61.39, 'Tc': 308.3, 'Tn': 189.4, 'Vc': 113.0, 'Zc': 0.271, 'omega': 0.187},
    'air': {'M': 28.851, 'Pc': 37.45, 'Tc': 132.2, 'Tn': None, 'Vc': 84.8, 'Zc': 0.289, 'omega': 0.035},
    'ammonia': {'M': 17.031, 'Pc': 112.8, 'Tc': 405.7, 'Tn': 239.7, 'Vc': 72.5, 'Zc': 0.242, 'omega': 0.253},
    'argon': {'M': 39.948, 'Pc': 48.98, 'Tc': 150.9, 'Tn': 87.3, 'Vc': 74.6, 'Zc': 0.291, 'omega': 0.0},
    'benzene': {'M': 78.114, 'Pc': 48.98, 'Tc': 562.2, 'Tn': 353.2, 'Vc': 259.0, 'Zc': 0.271, 'omega': 0.21},
    'benzoic acid': {'M': 122.123, 'Pc': 44.7, 'Tc': 751.0, 'Tn': 522.4, 'Vc': 344.0, 'Zc': 0.246, 'omega': 0.603},
    'biphenyl': {'M': 154.211, 'Pc': 38.5, 'Tc': 789.3, 'Tn': 528.2, 'Vc': 502.0, 'Zc': 0.295, 'omega': 0.365},
    'carbon dioxide': {'M': 44.01, 'Pc': 73.83, 'Tc': 304.2, 'Tn': None, 'Vc': 94.0, 'Zc': 0.274, 'omega': 0.224},
    'carbon disulfide': {'M': 76.143, 'Pc': 79.0, 'Tc': 552.0, 'Tn': 319.4, 'Vc': 160.0, 'Zc': 0.275, 'omega': 0.111},
    'carbon monoxide': {'M': 28.01, 'Pc': 34.99, 'Tc': 132.9, 'Tn': 81.7, 'Vc': 93.4, 'Zc': 0.299, 'omega': 0.048},
    'carbon tetrachloride': {'M': 153.822, 'Pc': 45.6, 'Tc': 556.4, 'Tn': 349.8, 'Vc': 276.0, 'Zc': 0.272, 'omega': 0.193},
    'chlorine': {'M': 70.905, 'Pc': 77.1, 'Tc': 417.2, 'Tn': 239.1, 'Vc': 124.0, 'Zc': 0.265, 'omega': 0.069},
    'chlorobenzene': {'M': 112.558, 'Pc': 45.2, 'Tc': 632.4, 'Tn': 404.9, 'Vc': 308.0, 'Zc': 0.265, 'omega': 0.25},
    'chloroform': {'M': 119.377, 'Pc': 54.72, 'Tc': 536.4, 'Tn': 334.3, 'Vc': 239.0, 'Zc': 0.293, 'omega': 0.222},
    'cis-2-butene': {'M': 56.108, 'Pc': 42.43, 'Tc': 435.6, 'Tn': 276.9, 'Vc': 233.8, 'Zc': 0.273, 'omega': 0.205},
    'cumene': {'M': 120.194, 'Pc': 32.09, 'Tc': 631.1, 'Tn': 425.6, 'Vc': 427.0, 'Zc': 0.261, 'omega': 0.326},
    'cyclohexane': {'M': 84.161, 'Pc': 40.73, 'Tc': 553.6, 'Tn': 353.9, 'Vc': 308.0, 'Zc': 0.273, 'omega': 0.21},
    'cyclohexene': {'M': 82.145, 'Pc': 43.5, 'Tc': 560.4, 'Tn': 356.1, 'Vc': 291.0, 'Zc': 0.272, 'omega': 0.212},
    'cyclopentane': {'M': 70.134, 'Pc': 45.02, 'Tc': 511.8, 'Tn': 322.4, 'Vc': 258.0, 'Zc': 0.273, 'omega': 0.196},
    'dichloromethane': {'M': 84.932, 'Pc': 60.8, 'Tc': 510.0, 'Tn': 312.9, 'Vc': 185.0, 'Zc': 0.265, 'omega': 0.199},
    'diethyl ether': {'M': 74.123, 'Pc': 36.4, 'Tc': 466.7, 'Tn': 307.6, 'Vc': 280.0, 'Zc': 0.263, 'omega': 0.281},
    'ethane': {'M': 30.07, 'Pc': 48.72, 'Tc': 305.3, 'Tn': 184.6, 'Vc': 145.5, 'Zc': 0.279, 'omega': 0.1},
    'ethanol': {'M': 46.069, 'Pc': 61.48, 'Tc': 513.9, 'Tn': 351.4, 'Vc': 167.0, 'Zc': 0.24, 'omega': 0.645},
    'ethyl acetate': {'M': 88.106, 'Pc': 38.8, 'Tc': 523.3, 'Tn': 350.2, 'Vc': 286.0, 'Zc': 0.255, 'omega': 0.366},
    'ethyl chloride': {'M': 64.514, 'Pc': 52.7, 'Tc': 460.4, 'Tn': 285.4, 'Vc': 200.0, 'Zc': 0.275, 'omega': 0.19},
    'ethylamine': {'M': 45.084, 'Pc': 56.2, 'Tc': 456.2, 'Tn': 289.7, 'Vc': 207.0, 'Zc': 0.307, 'omega': 0.285},
    'ethylbenzene': {'M': 106.167, 'Pc': 36.06, 'Tc': 617.2, 'Tn': 409.4, 'Vc': 374.0, 'Zc': 0.263, 'omega': 0.303},
    'ethylene': {'M': 28.054, 'Pc': 50.4, 'Tc': 282.3, 'Tn': 169.4, 'Vc': 131.0, 'Zc': 0.281, 'omega': 0.087},
    'ethylene glycol': {'M': 62.068, 'Pc': 77.0, 'Tc': 719.7, 'Tn': 470.5, 'Vc': 191.0, 'Zc': 0.246, 'omega': 0.487},
    'formaldehyde': {'M': 30.026, 'Pc': 65.9, 'Tc': 408.0, 'Tn': 254.1, 'Vc': 115.0, 'Zc': 0.223, 'omega': 0.282},
    'helium 4': {'M': 4.003, 'Pc': 2.28, 'Tc': 5.2, 'Tn': 4.2, 'Vc': 57.3, 'Zc': 0.302, 'omega': -0.39},
    'hydrogen': {'M': 2.016, 'Pc': 13.13, 'Tc': 33.19, 'Tn': 20.4, 'Vc': 64.1, 'Zc': 0.305, 'omega': -0.216},
    'hydrogen chloride': {'M': 36.461, 'Pc': 83.1, 'Tc': 324.7, 'Tn': 188.2, 'Vc': 81.0, 'Zc': 0.249, 'omega': 0.132},
    'hydrogen cyanide': {'M': 27.026, 'Pc': 53.9, 'Tc': 456.7, 'Tn': 298.9, 'Vc': 139.0, 'Zc': 0.197, 'omega': 0.41},
    'hydrogen sulfide': {'M': 34.082, 'Pc': 89.63, 'Tc': 373.5, 'Tn': 212.8, 'Vc': 98.5, 'Zc': 0.284, 'omega': 0.094},
    'isobutane': {'M': 58.123, 'Pc': 36.48, 'Tc': 408.1, 'Tn': 261.4, 'Vc': 262.7, 'Zc': 0.282, 'omega': 0.181},
    'isobutylene': {'M': 56.108, 'Pc': 40.0, 'Tc': 417.9, 'Tn': 266.3, 'Vc': 238.9, 'Zc': 0.275, 'omega': 0.194},
    'isooctane': {'M': 114.231, 'Pc': 25.68, 'Tc': 544.0, 'Tn': 372.4, 'Vc': 468.0, 'Zc': 0.266, 'omega': 0.302},
    'krypton': {'M': 83.8, 'Pc': 55.02, 'Tc': 209.4, 'Tn': 119.8, 'Vc': 91.2, 'Zc': 0.288, 'omega': 0.0},
    'm-xylene': {'M': 106.167, 'Pc': 35.36, 'Tc': 617.1, 'Tn': 412.3, 'Vc': 376.0, 'Zc': 0.259, 'omega': 0.326},
    'methane': {'M': 16.043, 'Pc': 45.99, 'Tc': 190.6, 'Tn': 111.4, 'Vc': 98.6, 'Zc': 0.286, 'omega': 0.012},
    'methanol': {'M': 32.042, 'Pc': 80.97, 'Tc': 512.6, 'Tn': 337.9, 'Vc': 118.0, 'Zc': 0.224, 'omega': 0.564},
    'methyl acetate': {'M': 74.079, 'Pc': 47.5, 'Tc': 506.6, 'Tn': 330.1, 'Vc': 228.0, 'Zc': 0.257, 'omega': 0.331},
    'methyl chloride': {'M': 50.488, 'Pc': 66.8, 'Tc': 416.3, 'Tn': 249.1, 'Vc': 143.0, 'Zc': 0.276, 'omega': 0.153},
    'methyl ethyl ketone': {'M': 72.107, 'Pc': 41.5, 'Tc': 535.5, 'Tn': 352.8, 'Vc': 267.0, 'Zc': 0.249, 'omega': 0.323},
    'methyl t-butyl ether': {'M': 88.15, 'Pc': 34.3, 'Tc': 497.1, 'Tn': 328.4, 'Vc': 329.0, 'Zc': 0.273, 'omega': 0.266},
    'methylamine': {'M': 31.057, 'Pc': 74.6, 'Tc': 430.1, 'Tn': 266.8, 'Vc': 154.0, 'Zc': 0.321, 'omega': 0.281},
    'methylcyclohexane': {'M': 98.188, 'Pc': 34.71, 'Tc': 572.2, 'Tn': 374.1, 'Vc': 368.0, 'Zc': 0.269, 'omega': 0.235},
    'methylcyclopentane': {'M': 84.161, 'Pc': 37.85, 'Tc': 532.8, 'Tn': 345.0, 'Vc': 319.0, 'Zc': 0.272, 'omega': 0.23},
    'n-butane': {'M': 58.123, 'Pc': 37.96, 'Tc': 425.1, 'Tn': 272.7, 'Vc': 255.0, 'Zc': 0.274, 'omega': 0.2},
    'n-butyric acid': {'M': 88.106, 'Pc': 40.64, 'Tc': 615.7, 'Tn': 436.4, 'Vc': 291.7, 'Zc': 0.232, 'omega': 0.681},
    'n-decane': {'M': 142.285, 'Pc': 21.1, 'Tc': 617.7, 'Tn': 447.3, 'Vc': 600.0, 'Zc': 0.247, 'omega': 0.492},
    'n-heptane': {'M': 100.204, 'Pc': 27.4, 'Tc': 540.2, 'Tn': 371.6, 'Vc': 428.0, 'Zc': 0.261, 'omega': 0.35},
    'n-hexane': {'M': 86.177, 'Pc': 30.25, 'Tc': 507.6, 'Tn': 341.9, 'Vc': 371.0, 'Zc': 0.266, 'omega': 0.301},
    'n-nonane': {'M': 128.258, 'Pc': 22.9, 'Tc': 594.6, 'Tn': 424.0, 'Vc': 544.0, 'Zc': 0.252, 'omega': 0.444},
    'n-octane': {'M': 114.231, 'Pc': 24.9, 'Tc': 568.7, 'Tn': 398.8, 'Vc': 486.0, 'Zc': 0.256, 'omega': 0.4},
    'n-pentane': {'M': 72.15, 'Pc': 33.7, 'Tc': 469.7, 'Tn': 309.2, 'Vc': 313.0, 'Zc': 0.27, 'omega': 0.252},
    'naphthalene': {'M': 128.174, 'Pc': 40.51, 'Tc': 748.4, 'Tn': 491.2, 'Vc': 413.0, 'Zc': 0.269, 'omega': 0.302},
    'nitric acid': {'M': 63.013, 'Pc': 68.9, 'Tc': 520.0, 'Tn': 356.2, 'Vc': 145.0, 'Zc': 0.231, 'omega': 0.714},
    'nitric oxide (no)': {'M': 30.006, 'Pc': 64.8, 'Tc': 180.2, 'Tn': 121.4, 'Vc': 58.0, 'Zc': 0.251, 'omega': 0.583},
    'nitrogen': {'M': 28.014, 'Pc': 34.0, 'Tc': 126.2, 'Tn': 77.3, 'Vc': 89.2, 'Zc': 0.289, 'omega': 0.038},
    'nitromethane': {'M': 61.04, 'Pc': 63.1, 'Tc': 588.2, 'Tn': 374.4, 'Vc': 173.0, 'Zc': 0.223, 'omega': 0.348},
    'nitrous oxide (n2o)': {'M': 44.013, 'Pc': 72.45, 'Tc': 309.6, 'Tn': 184.7, 'Vc': 97.4, 'Zc': 0.274, 'omega': 0.141},
    'o-xylene': {'M': 106.167, 'Pc': 37.34, 'Tc': 630.3, 'Tn': 417.6, 'Vc': 369.0, 'Zc': 0.263, 'omega': 0.31},
    'oxygen': {'M': 31.999, 'Pc': 50.43, 'Tc': 154.6, 'Tn': 90.2, 'Vc': 73.4, 'Zc': 0.288, 'omega': 0.022},
    'p-xylene': {'M': 106.167, 'Pc': 35.11, 'Tc': 616.2, 'Tn': 411.5, 'Vc': 379.0, 'Zc': 0.26, 'omega': 0.322},
    'phenol': {'M': 94.113, 'Pc': 61.3, 'Tc': 694.3, 'Tn': 455.0, 'Vc': 229.0, 'Zc': 0.243, 'omega': 0.444},
    'propane': {'M': 44.097, 'Pc': 42.48, 'Tc': 369.8, 'Tn': 231.1, 'Vc': 200.0, 'Zc': 0.276, 'omega': 0.152},
    'propylene': {'M': 42.081, 'Pc': 46.65, 'Tc': 365.6, 'Tn': 225.5, 'Vc': 188.4, 'Zc': 0.289, 'omega': 0.14},
    'styrene': {'M': 104.152, 'Pc': 38.4, 'Tc': 636.0, 'Tn': 418.3, 'Vc': 352.0, 'Zc': 0.256, 'omega': 0.297},
    'sulfur dioxide': {'M': 64.065, 'Pc': 78.84, 'Tc': 430.8, 'Tn': 263.1, 'Vc': 122.0, 'Zc': 0.269, 'omega': 0.245},
    'sulfur trioxide': {'M': 80.064, 'Pc': 82.1, 'Tc': 490.9, 'Tn': 317.9, 'Vc': 127.0, 'Zc': 0.255, 'omega': 0.424},
    'sulfuric acid': {'M': 98.08, 'Pc': 64.0, 'Tc': 924.0, 'Tn': 610.0, 'Vc': 177.0, 'Zc': 0.147, 'omega': None},
    'tetrafluoroethane': {'M': 102.03, 'Pc': 40.6, 'Tc': 374.2, 'Tn': 247.1, 'Vc': 198.0, 'Zc': 0.258, 'omega': 0.327},
    'toluene': {'M': 92.141, 'Pc': 41.06, 'Tc': 591.8, 'Tn': 383.8, 'Vc': 316.0, 'Zc': 0.264, 'omega': 0.262},
    'trans-2-butene': {'M': 56.108, 'Pc': 41.0, 'Tc': 428.6, 'Tn': 274.0, 'Vc': 237.7, 'Zc': 0.275, 'omega': 0.218},
    'water': {'M': 18.015, 'Pc': 220.55, 'Tc': 647.1, 'Tn': 373.2, 'Vc': 55.9, 'Zc': 0.229, 'omega': 0.345},
    'xenon': {'M': 131.3, 'Pc': 58.4, 'Tc': 289.7, 'Tn': 165.0, 'Vc': 118.0, 'Zc': 0.286, 'omega': 0.0},
}

_C1 = {
    '1,3-butadiene': (2.734, 0.026786, -8.882e-06, 0.0, 1500.0),
    '1-butene': (1.967, 0.03163, -9.873e-06, 0.0, 1500.0),
    '1-heptene': (3.768, 0.056588, -1.7847e-05, 0.0, 1500.0),
    '1-hexene': (3.22, 0.048189, -1.5157e-05, 0.0, 1500.0),
    '1-octene': (4.324, 0.06496, -2.0521e-05, 0.0, 1500.0),
    '1-pentene': (2.691, 0.039753, -1.2447e-05, 0.0, 1500.0),
    'acetaldehyde': (1.693, 0.017978, -6.158e-06, 0.0, 1000.0),
    'acetylene': (6.132, 0.001952, 0.0, -129900.0, 1500.0),
    'air': (3.355, 0.000575, 0.0, -1600.0, 2000.0),
    'ammonia': (3.578, 0.00302, 0.0, -18600.0, 1800.0),
    'benzene': (-0.206, 0.039064, -1.3301e-05, 0.0, 1500.0),
    'bromine': (4.493, 5.6e-05, 0.0, -15400.0, 3000.0),
    'carbon dioxide': (5.457, 0.001045, 0.0, -115700.0, 2000.0),
    'carbon disulfide': (6.311, 0.000805, 0.0, -90600.0, 1800.0),
    'carbon monoxide': (3.376, 0.000557, 0.0, -3100.0, 2500.0),
    'chlorine': (4.442, 8.9e-05, 0.0, -34400.0, 3000.0),
    'cyclohexane': (-3.876, 0.063249, -2.0928e-05, 0.0, 1500.0),
    'dinitrogen tetroxide': (11.66, 0.002257, 0.0, -278700.0, 2000.0),
    'ethane': (1.131, 0.019225, -5.561e-06, 0.0, 1500.0),
    'ethanol': (3.518, 0.020001, -6.002e-06, 0.0, 1500.0),
    'ethylbenzene': (1.124, 0.05538, -1.8476e-05, 0.0, 1500.0),
    'ethylene': (1.424, 0.014394, -4.392e-06, 0.0, 1500.0),
    'ethylene oxide': (-0.385, 0.023463, -9.296e-06, 0.0, 1000.0),
    'formaldehyde': (2.264, 0.007022, -1.877e-06, 0.0, 1500.0),
    'hydrogen': (3.249, 0.000422, 0.0, 8300.0, 3000.0),
    'hydrogen chloride': (3.156, 0.000623, 0.0, 15100.0, 2000.0),
    'hydrogen cyanide': (4.736, 0.001359, 0.0, -72500.0, 2500.0),
    'hydrogen sulfide': (3.931, 0.00149, 0.0, -23200.0, 2300.0),
    'isobutane': (1.677, 0.037853, -1.1945e-05, 0.0, 1500.0),
    'methane': (1.702, 0.009081, -2.164e-06, 0.0, 1500.0),
    'methanol': (2.211, 0.012216, -3.45e-06, 0.0, 1500.0),
    'n-butane': (1.935, 0.036915, -1.1402e-05, 0.0, 1500.0),
    'n-heptane': (3.57, 0.062127, -1.9486e-05, 0.0, 1500.0),
    'n-hexane': (3.025, 0.053722, -1.6791e-05, 0.0, 1500.0),
    'n-octane': (4.108, 0.070567, -2.2208e-05, 0.0, 1500.0),
    'n-pentane': (2.464, 0.045351, -1.4111e-05, 0.0, 1500.0),
    'nitric oxide (no)': (3.387, 0.000629, 0.0, 1400.0, 2000.0),
    'nitrogen': (3.28, 0.000593, 0.0, 4000.0, 2000.0),
    'nitrogen dioxide': (4.982, 0.001195, 0.0, -79200.0, 2000.0),
    'nitrous oxide (n2o)': (5.328, 0.001214, 0.0, -92800.0, 2000.0),
    'oxygen': (3.639, 0.000506, 0.0, -22700.0, 2000.0),
    'propane': (1.213, 0.028785, -8.824e-06, 0.0, 1500.0),
    'propylene': (1.637, 0.022706, -6.915e-06, 0.0, 1500.0),
    'styrene': (2.05, 0.050192, -1.6662e-05, 0.0, 1500.0),
    'sulfur dioxide': (5.699, 0.000801, 0.0, -101500.0, 2000.0),
    'sulfur trioxide': (8.06, 0.001056, 0.0, -202800.0, 2000.0),
    'toluene': (0.29, 0.047052, -1.5716e-05, 0.0, 1500.0),
    'water': (3.47, 0.00145, 0.0, 12100.0, 2000.0),
}

_ALIASES = {
    "butane": "n-butane", "nbutane": "n-butane", "n butane": "n-butane",
    "iso-butane": "isobutane", "i-butane": "isobutane",
    "pentane": "n-pentane", "hexane": "n-hexane", "heptane": "n-heptane",
    "octane": "n-octane", "n2": "nitrogen", "o2": "oxygen", "h2": "hydrogen",
    "co": "carbon monoxide", "co2": "carbon dioxide", "h2s": "hydrogen sulfide",
    "so2": "sulfur dioxide", "ar": "argon", "cl2": "chlorine", "he": "helium",
    "nh3": "ammonia", "h2o": "water", "ch4": "methane", "c2h6": "ethane",
    "c3h8": "propane", "c2h4": "ethylene", "c2h2": "acetylene",
    "ch3oh": "methanol", "c2h5oh": "ethanol",
    "helium": "helium 4", "nitric oxide": "nitric oxide (no)", "no": "nitric oxide (no)",
    "n2o": "nitrous oxide (n2o)", "nitrous oxide": "nitrous oxide (n2o)",
    "iso-butane": "isobutane", "hcl": "hydrogen chloride", "hcn": "hydrogen cyanide",
    "ccl4": "carbon tetrachloride", "mtbe": "methyl t-butyl ether", "so3": "sulfur trioxide",
    "cs2": "carbon disulfide", "nh4": "ammonia", "1-butene": "1-butene",
}


@dataclass(frozen=True)
class Species:
    name: str
    Tc: float      # K
    Pc: float      # bar
    omega: float
    M: Optional[float] = None  # g/mol
    cp: Optional[tuple] = None  # (A, B, C, D) for ideal-gas Cp/R; needed for total H and S

    @staticmethod
    def lookup(name: str) -> "Species":
        key = " ".join(name.strip().lower().replace("_", " ").split())
        key = _ALIASES.get(key, key)
        if key not in _B1 and key.replace("-", " ") in _B1:   # "carbon-dioxide" -> "carbon dioxide"
            key = key.replace("-", " ")
        if key not in _B1:
            raise KeyError(
                f"Unknown species '{name}'. Use list-species, or supply "
                f"--Tc/--Pc/--omega for a custom fluid."
            )
        d = _B1[key]
        if d["omega"] is None:
            raise KeyError(f"'{name}': no acentric factor in Table B.1; supply --omega.")
        cp = _C1.get(key)
        return Species(key, d["Tc"], d["Pc"], d["omega"], d["M"], tuple(cp[:4]) if cp else None)


def list_species() -> List[str]:
    return sorted(_B1)


# ----------------------------------------------------------------------------
# Ideal-gas property changes from Cp^ig/R = A + B T + C T^2 + D T^-2
# ----------------------------------------------------------------------------
R_J = R_GAS * 0.1  # J/(mol K)


def ig_delta_H(cp, T0: float, T: float) -> float:
    """H^ig(T) - H^ig(T0)  [J/mol]"""
    A, B, C, D = cp
    return R_J * (A * (T - T0) + B / 2 * (T ** 2 - T0 ** 2) + C / 3 * (T ** 3 - T0 ** 3)
                  - D * (1.0 / T - 1.0 / T0))


def ig_delta_S(cp, T0: float, T: float, P0: float, P: float) -> float:
    """S^ig(T,P) - S^ig(T0,P0)  [J/(mol K)]"""
    A, B, C, D = cp
    return R_J * (A * math.log(T / T0) + B * (T - T0) + C / 2 * (T ** 2 - T0 ** 2)
                  - D / 2 * (T ** -2 - T0 ** -2) - math.log(P / P0))


# ----------------------------------------------------------------------------
# Core model
# ----------------------------------------------------------------------------
@dataclass
class StateResult:
    species: str
    eos: str
    T: float
    P: float
    phase: str
    Z: float
    V: float                 # cm^3/mol
    rho: float               # mol/L
    Z_roots: List[float]
    V_ideal: float
    dev_ideal_pct: float     # (V - Vig)/V * 100
    ln_phi: float
    phi: float
    Psat: Optional[float] = None
    Vsat_liq: Optional[float] = None
    Vsat_vap: Optional[float] = None
    mass_density: Optional[float] = None  # kg/m^3
    notes: List[str] = field(default_factory=list)
    H_res: Optional[float] = None   # residual enthalpy  H - H^ig  [J/mol]
    S_res: Optional[float] = None   # residual entropy   S - S^ig  [J/(mol K)]
    H: Optional[float] = None       # total enthalpy rel. to ideal gas at (T0, P0) [J/mol]
    S: Optional[float] = None       # total entropy  rel. to ideal gas at (T0, P0) [J/(mol K)]
    ref: Optional[tuple] = None     # (T0, P0) reference


class CubicEOS:
    """Generic cubic EOS bound to a species."""

    def __init__(self, species: Species, eos: str = "PR"):
        if eos not in EOS_MODELS:
            raise ValueError(f"eos must be one of {list(EOS_MODELS)}")
        self.sp = species
        self.eos = EOS_MODELS[eos]
        self.R = R_GAS
        self.T_ref, self.P_ref = 298.15, 1.0   # H = S = 0 for ideal gas here

    # -- parameters ----------------------------------------------------------
    @property
    def b(self) -> float:
        """Eq. 3.44: b = Omega R Tc / Pc  [cm^3/mol]"""
        return self.eos.Omega * self.R * self.sp.Tc / self.sp.Pc

    def a(self, T: float) -> float:
        """Eq. 3.45: a(T) = Psi alpha(Tr;w) R^2 Tc^2 / Pc  [bar cm^6/mol^2]"""
        Tr = T / self.sp.Tc
        return (self.eos.Psi * self.eos.alpha(Tr, self.sp.omega)
                * self.R ** 2 * self.sp.Tc ** 2 / self.sp.Pc)

    def beta_q(self, T: float, P: float):
        """Eqs. 3.50-3.51"""
        Tr, Pr = T / self.sp.Tc, P / self.sp.Pc
        beta = self.eos.Omega * Pr / Tr
        q = self.eos.Psi * self.eos.alpha(Tr, self.sp.omega) / (self.eos.Omega * Tr)
        return beta, q

    # -- direct P(T,V) -------------------------------------------------------
    def pressure(self, T: float, V: float) -> float:
        """Eq. 3.41 evaluated directly [bar]."""
        T, V = float(T), float(V)
        b = self.b
        if V <= b:
            raise ValueError(f"V = {V:g} must exceed b = {b:.4g} cm^3/mol")
        e, s = self.eos.eps, self.eos.sigma
        return self.R * T / (V - b) - self.a(T) / ((V + e * b) * (V + s * b))

    # -- Z roots -------------------------------------------------------------
    def Z_roots(self, T: float, P: float) -> List[float]:
        """All physically meaningful (real, Z > beta) roots, ascending.

        Cubic from Eq. 3.48:
        Z^3 + [(e+s)B - (1+B)] Z^2 + [e s B^2 - (1+B)(e+s)B + qB] Z
              - [(1+B) e s B^2 + q B^2] = 0       (B = beta)
        """
        T, P = float(T), float(P)
        if T <= 0 or P <= 0:
            raise ValueError("T and P must be positive")
        B, q = self.beta_q(T, P)
        e, s = self.eos.eps, self.eos.sigma
        c2 = (e + s) * B - (1.0 + B)
        c1 = e * s * B ** 2 - (1.0 + B) * (e + s) * B + q * B
        c0 = -((1.0 + B) * e * s * B ** 2 + q * B ** 2)
        roots = np.roots([1.0, c2, c1, c0])
        good = []
        for r in roots:
            if abs(r.imag) < 1e-4 * max(1.0, abs(r.real)) and r.real > B * (1 + 1e-12):
                # polish with Newton on the real cubic
                z0 = z = r.real
                for _ in range(20):          # Newton polish (skipped near multiple roots)
                    f = ((z + c2) * z + c1) * z + c0
                    df = (3 * z + 2 * c2) * z + c1
                    if abs(df) < 1e-8:
                        break
                    dz = f / df
                    z -= dz
                    if abs(dz) < 1e-15:
                        break
                if abs(z - z0) > 1e-6 or z <= B:
                    z = z0
                good.append(z)
        good.sort()
        dedup: List[float] = []
        for z in good:
            if not dedup or abs(z - dedup[-1]) > 1e-6 * max(1.0, z):
                dedup.append(z)
        return dedup

    def volume_roots(self, T: float, P: float) -> List[float]:
        """Molar-volume roots [cm^3/mol], ascending (V = Z R T / P)."""
        return [z * self.R * T / P for z in self.Z_roots(T, P)]

    # -- fugacity ------------------------------------------------------------
    def ln_phi(self, T: float, P: float, Z: float) -> float:
        """ln(phi) = Z - 1 - ln(Z - beta) - q I   (I per Eq. 13.x, generic cubic)."""
        B, q = self.beta_q(T, P)
        e, s = self.eos.eps, self.eos.sigma
        if abs(s - e) < 1e-14:
            I = B / (Z + e * B)
        else:
            I = math.log((Z + s * B) / (Z + e * B)) / (s - e)
        return Z - 1.0 - math.log(Z - B) - q * I

    # -- thermodynamic properties (H, S) --------------------------------------
    def dlnalpha_dlnTr(self, T: float) -> float:
        """d ln alpha / d ln Tr (analytic)."""
        name, w, Tr = self.eos.name, self.sp.omega, float(T) / self.sp.Tc
        if name == "vdW":
            return 0.0
        if name == "RK":
            return -0.5
        kappa = (0.480 + 1.574 * w - 0.176 * w * w) if name == "SRK" else \
                (0.37464 + 1.54226 * w - 0.26992 * w * w)
        rt = math.sqrt(Tr)
        return -kappa * rt / (1.0 + kappa * (1.0 - rt))

    def _I(self, Z: float, beta: float) -> float:
        e, s = self.eos.eps, self.eos.sigma
        if abs(s - e) < 1e-14:
            return beta / (Z + e * beta)
        return math.log((Z + s * beta) / (Z + e * beta)) / (s - e)

    def residual(self, T: float, P: float, Z: float):
        """Residual enthalpy [J/mol] and entropy [J/(mol K)] for root Z.

            H^R/RT = Z - 1 + (dln(alpha)/dln(Tr) - 1) q I
            S^R/R  = ln(Z - beta) + (dln(alpha)/dln(Tr)) q I
        (Smith/Van Ness Eqs. 6.67-6.68; reference = ideal gas at same T, P.)
        """
        T, P = float(T), float(P)
        beta, q = self.beta_q(T, P)
        I = self._I(Z, beta)
        d = self.dlnalpha_dlnTr(T)
        HR = R_J * T * (Z - 1.0 + (d - 1.0) * q * I)
        SR = R_J * (math.log(Z - beta) + d * q * I)
        return HR, SR

    def total_HS(self, T: float, P: float, Z: float):
        """(H, S) relative to ideal gas at (T_ref, P_ref); needs species.cp."""
        if self.sp.cp is None:
            return None, None
        HR, SR = self.residual(T, P, Z)
        H = ig_delta_H(self.sp.cp, self.T_ref, T) + HR
        S = ig_delta_S(self.sp.cp, self.T_ref, T, self.P_ref, P) + SR
        return H, S

    def vaporization(self, T: float) -> dict:
        """Saturation properties at T < Tc, including latent heat and entropy of vaporization."""
        T = float(T)
        Ps, Vl, Vv = self.saturation(T)
        zl, zv = Vl * Ps / (self.R * T), Vv * Ps / (self.R * T)
        HRl, SRl = self.residual(T, Ps, zl)
        HRv, SRv = self.residual(T, Ps, zv)
        out = dict(T=T, Psat=Ps, V_liq=Vl, V_vap=Vv, Z_liq=zl, Z_vap=zv,
                   HR_liq=HRl, HR_vap=HRv, SR_liq=SRl, SR_vap=SRv,
                   dH_vap=HRv - HRl, dS_vap=SRv - SRl)
        Hl, Sl = self.total_HS(T, Ps, zl)
        Hv, Sv = self.total_HS(T, Ps, zv)
        out.update(H_liq=Hl, H_vap=Hv, S_liq=Sl, S_vap=Sv)
        return out

    def delta(self, T1: float, P1: float, T2: float, P2: float, phase1=None, phase2=None) -> dict:
        """dH, dS between two states (same species, may cross a phase boundary).
        Requires species.cp. phase = None (stable), 'liquid' or 'vapor' to force a root."""
        s1 = self.state(T1, P1, phase1)
        s2 = self.state(T2, P2, phase2)
        if s1.H is None:
            raise ValueError("Ideal-gas Cp data missing for this species (supply --cp A B C D).")
        return dict(state1=s1, state2=s2, dH=s2.H - s1.H, dS=s2.S - s1.S)

    # -- saturation ----------------------------------------------------------
    def _spinodal_pressures(self, T: float):
        """(P_min, P_max) of the van der Waals loop at T < Tc, or None."""
        T = float(T)
        b, a, RT = float(self.b), float(self.a(T)), float(self.R * T)
        e, s = self.eos.eps, self.eos.sigma
        V = np.poly1d([1.0, 0.0])
        poly = RT * (V + e * b) ** 2 * (V + s * b) ** 2 - a * (2 * V + (e + s) * b) * (V - b) ** 2
        rts = [r.real for r in poly.roots
               if abs(r.imag) < 1e-9 * max(1, abs(r.real)) and r.real > b * (1 + 1e-9)]
        rts.sort()
        if len(rts) < 2:
            return None
        P1, P2 = self.pressure(T, rts[0]), self.pressure(T, rts[-1])
        return min(P1, P2), max(P1, P2)

    def saturation(self, T: float):
        """Return (Psat, V_liq, V_vap) at T < Tc from equal fugacity."""
        T = float(T)
        if T >= self.sp.Tc:
            raise ValueError(f"T = {T:g} K is not below Tc = {self.sp.Tc:g} K; no saturation.")
        sp = self._spinodal_pressures(T)
        if sp is None:
            raise RuntimeError("No van der Waals loop found at this T.")
        Pmin, Pmax = sp
        hi = Pmax * (1 - 1e-9)
        lo = max(Pmin, 0.0) * (1 + 1e-9)
        if lo <= 0:
            lo = 1e-12 * self.sp.Pc
            for _ in range(40):
                if len(self.Z_roots(T, lo)) >= 3:
                    break
                lo *= 10

        def g(P):  # ln phi_L - ln phi_V ; >0 => vapour stable
            zs = self.Z_roots(T, P)
            return self.ln_phi(T, P, zs[0]) - self.ln_phi(T, P, zs[-1])

        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if g(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi / lo - 1 < 1e-13:
                break
        Ps = math.sqrt(lo * hi)
        zs = self.Z_roots(T, Ps)
        return Ps, zs[0] * self.R * T / Ps, zs[-1] * self.R * T / Ps

    # -- full state ----------------------------------------------------------
    def state(self, T: float, P: float, phase: Optional[str] = None) -> StateResult:
        """State at (T, P) with diagnostics. phase=None picks the stable phase;
        'liquid' / 'vapor' force the smallest / largest root (metastable states allowed)."""
        T, P = float(T), float(P)
        zs = self.Z_roots(T, P)
        notes: List[str] = []
        Psat = Vl = Vv = None
        Tr = T / self.sp.Tc
        if Tr < 1.0:
            Psat, Vl, Vv = self.saturation(T)
            if abs(P - Psat) / Psat < 1e-6:
                phase_auto, Z = "saturated (L+V)", zs[-1]
                notes.append("At saturation: vapour root returned; liquid root also listed.")
            elif P < Psat:
                phase_auto, Z = "vapour", zs[-1]
            else:
                phase_auto, Z = "liquid", zs[0]
        else:
            phase_auto, Z = ("supercritical fluid" if (Tr > 1 or P > self.sp.Pc) else "critical"), zs[-1]
            if Tr > 1 and P < self.sp.Pc:
                phase_auto = "gas (T > Tc)"
        if phase in ("liquid", "vapor", "vapour"):
            Z = zs[0] if phase == "liquid" else zs[-1]
            phase = phase.replace("vapor", "vapour") + " (forced)"
            if Psat is not None and ((phase.startswith("liquid") and P < Psat) or
                                     (phase.startswith("vapour") and P > Psat)):
                notes.append("Forced phase is metastable at this T, P.")
        elif phase is not None:
            raise ValueError("phase must be None, 'liquid' or 'vapor'")
        else:
            phase = phase_auto
        V = Z * self.R * T / P
        Vig = self.R * T / P
        lnp = self.ln_phi(T, P, Z)
        HR, SR = self.residual(T, P, Z)
        Htot, Stot = self.total_HS(T, P, Z)
        if phase.startswith("liquid") and self.eos.name in ("vdW", "RK", "SRK"):
            notes.append("Liquid-volume accuracy of vdW/RK/SRK is poor; PR or "
                         "Rackett/Lydersen correlations are preferable.")
        if self.sp.omega < 0:
            notes.append("Negative acentric factor (quantum gas): treat results with caution; "
                         "consider effective critical constants (Eqs. 3.54-3.56).")
        rho_m = None
        if self.sp.M:
            rho_m = self.sp.M / V * 1000.0  # g/cm3 *1000 = kg/m3
        return StateResult(
            self.sp.name, self.eos.name, T, P, phase, Z, V, 1000.0 / V, zs, Vig,
            (V - Vig) / V * 100.0, lnp, math.exp(lnp), Psat, Vl, Vv, rho_m, notes,
            HR, SR, Htot, Stot, (self.T_ref, self.P_ref) if Htot is not None else None)

    # -- inverse problems ----------------------------------------------------
    def temperature(self, P: float, V: float) -> float:
        """T [K] giving pressure P at molar volume V (bisection; P(T,V) increases with T)."""
        f = lambda T: self.pressure(T, V) - P
        lo, hi = 0.05 * self.sp.Tc, 60.0 * self.sp.Tc
        if f(lo) > 0 or f(hi) < 0:
            raise ValueError("No temperature in [0.05 Tc, 60 Tc] satisfies the (P, V) pair.")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-10:
                break
        return 0.5 * (lo + hi)

    # -- isotherm ------------------------------------------------------------
    def isotherm(self, T: float, Vmin: Optional[float] = None, Vmax: Optional[float] = None,
                 n: int = 200):
        """Return arrays (V, P) along an isotherm (raw EOS, includes the vdW loop)."""
        Vmin = Vmin or 1.02 * self.b
        Vmax = Vmax or 60.0 * self.R * self.sp.Tc / self.sp.Pc
        V = np.geomspace(Vmin, Vmax, n)
        return V, np.array([self.pressure(T, v) for v in V])


# ----------------------------------------------------------------------------
# Cross-check: Pitzer second-virial correlation (Eqs. 3.57-3.62)
# ----------------------------------------------------------------------------
def virial_pitzer_Z(sp: Species, T: float, P: float) -> float:
    Tr, Pr = T / sp.Tc, P / sp.Pc
    B0 = 0.083 - 0.422 / Tr ** 1.6
    B1 = 0.139 - 0.172 / Tr ** 4.2
    return 1.0 + (B0 + sp.omega * B1) * Pr / Tr


# ----------------------------------------------------------------------------
# Unit handling
# ----------------------------------------------------------------------------
_P_TO_BAR = {"bar": 1.0, "kpa": 0.01, "mpa": 10.0, "pa": 1e-5, "atm": 1.01325, "psi": 0.0689476}
_V_TO_CM3 = {"cm3/mol": 1.0, "l/mol": 1000.0, "m3/kmol": 1000.0, "m3/mol": 1e6}


def T_to_K(x: float, u: str) -> float:
    u = u.lower()
    if u == "k":
        return x
    if u == "c":
        return x + 273.15
    if u == "f":
        return (x - 32.0) * 5.0 / 9.0 + 273.15
    raise ValueError("T unit must be K, C or F")


def T_from_K(x: float, u: str) -> float:
    u = u.lower()
    return {"k": x, "c": x - 273.15, "f": (x - 273.15) * 9 / 5 + 32}[u]


def P_to_bar(x, u):
    return x * _P_TO_BAR[u.lower()]


def P_from_bar(x, u):
    return x / _P_TO_BAR[u.lower()]


def V_to_cm3(x, u):
    return x * _V_TO_CM3[u.lower()]


def V_from_cm3(x, u):
    return x / _V_TO_CM3[u.lower()]


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_pv(sp: Species, eos_name: str, Tr_list=(0.7, 0.8, 0.9, 1.0, 1.1, 1.3), out="pv_diagram.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = CubicEOS(sp, eos_name)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for Tr in Tr_list:
        T = Tr * sp.Tc
        V, P = m.isotherm(T, n=600)
        ax.plot(V, P, lw=1.4, label=f"Tr = {Tr:g}  ({T:.1f} K)")
    # saturation dome
    Vl, Vv, Ps = [], [], []
    for Tr in np.linspace(0.5, 0.9995, 120):
        try:
            p, vl, vv = m.saturation(Tr * sp.Tc)
            Ps.append(p); Vl.append(vl); Vv.append(vv)
        except Exception:
            pass
    ax.plot(Vl, Ps, "k-", lw=2)
    ax.plot(Vv, Ps, "k-", lw=2, label="saturation dome")
    ax.set_xscale("log")
    ax.set_xlim(1.0 * m.b, 40 * R_GAS * sp.Tc / sp.Pc)
    ax.set_ylim(0, 2.5 * sp.Pc)
    ax.set_xlabel("V  [cm³/mol]")
    ax.set_ylabel("P  [bar]")
    ax.set_title(f"{sp.name}: {eos_name} isotherms")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_z(sp: Species, T: float, Pmax: float, out="z_vs_p.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    Ps = np.linspace(Pmax / 400, Pmax, 400)
    for name in EOS_MODELS:
        m = CubicEOS(sp, name)
        Z = []
        for P in Ps:
            try:
                Z.append(m.state(T, P).Z)
            except Exception:
                Z.append(np.nan)
        ax.plot(Ps, Z, label=name)
    ax.plot(Ps, [virial_pitzer_Z(sp, T, P) for P in Ps], "k--", lw=1, label="Pitzer virial (B)")
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("P  [bar]")
    ax.set_ylabel("Z")
    ax.set_title(f"{sp.name} at {T:.2f} K - compressibility factor")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _species_from_args(a) -> Species:
    cp = tuple(a.cp) if getattr(a, "cp", None) else None
    if a.Tc and a.Pc and a.omega is not None:
        return Species(a.species or "custom", a.Tc, a.Pc, a.omega, a.M, cp)
    sp = Species.lookup(a.species)
    if a.Tc or a.Pc or a.omega is not None or cp:
        sp = Species(sp.name, a.Tc or sp.Tc, a.Pc or sp.Pc,
                     sp.omega if a.omega is None else a.omega, a.M or sp.M, cp or sp.cp)
    return sp


def _set_ref(m: "CubicEOS", a):
    if getattr(a, "T0", None) is not None:
        m.T_ref = T_to_K(a.T0, a.T_unit)
    if getattr(a, "P0", None) is not None:
        m.P_ref = P_to_bar(a.P0, a.P_unit)


def _print_state(s: StateResult, a):
    Tu, Pu, Vu = a.T_unit, a.P_unit, a.V_unit
    print(f"\n{s.species} | {s.eos} | T = {T_from_K(s.T, Tu):.4g} {Tu.upper()} "
          f"({s.T:.2f} K), P = {P_from_bar(s.P, Pu):.5g} {Pu}")
    print(f"  Phase              : {s.phase}")
    print(f"  Z                  : {s.Z:.5f}")
    print(f"  V                  : {V_from_cm3(s.V, Vu):.6g} {Vu}")
    print(f"  Density            : {s.rho:.5g} mol/L" +
          (f"  ({s.mass_density:.5g} kg/m3)" if s.mass_density else ""))
    print(f"  Ideal-gas V        : {V_from_cm3(s.V_ideal, Vu):.6g} {Vu}  "
          f"(ideal gas error {-s.dev_ideal_pct:+.2f}% on V)")
    print(f"  Fugacity coeff.    : {s.phi:.5f}  (f = {P_from_bar(s.phi * s.P, Pu):.5g} {Pu})")
    print(f"  H^R (residual)     : {s.H_res:.6g} J/mol      S^R : {s.S_res:.6g} J/(mol K)")
    if s.H is not None:
        print(f"  H (total)          : {s.H:.6g} J/mol      S   : {s.S:.6g} J/(mol K)"
              f"   [ref: ideal gas at {s.ref[0]:.2f} K, {s.ref[1]:g} bar, H = S = 0]")
    if s.Psat is not None:
        print(f"  Psat(T)            : {P_from_bar(s.Psat, Pu):.5g} {Pu};  "
              f"V_liq = {V_from_cm3(s.Vsat_liq, Vu):.5g}, V_vap = {V_from_cm3(s.Vsat_vap, Vu):.5g} {Vu}")
    print("  Z roots            : " + ", ".join(f"{z:.5f}" for z in s.Z_roots))
    for n in s.notes:
        print(f"  note: {n}")


def build_parser():
    p = argparse.ArgumentParser(description="PVT of real fluids from cubic equations of state.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, need_eos=True):
        sp.add_argument("species", nargs="?", help="species name/formula (see list-species)")
        sp.add_argument("--Tc", type=float, help="critical T [K] (custom/override)")
        sp.add_argument("--Pc", type=float, help="critical P [bar] (custom/override)")
        sp.add_argument("--omega", type=float, help="acentric factor (custom/override)")
        sp.add_argument("--M", type=float, help="molar mass g/mol (optional)")
        sp.add_argument("--cp", type=float, nargs=4, metavar=("A", "B", "C", "D"),
                        help="ideal-gas Cp/R = A + B T + C T^2 + D T^-2 (T in K); overrides Table C.1")
        sp.add_argument("--T0", type=float, help="reference T for H = S = 0 (ideal gas), default 298.15 K")
        sp.add_argument("--P0", type=float, help="reference P for H = S = 0 (ideal gas), default 1 bar")
        if need_eos:
            sp.add_argument("--eos", default="PR", choices=list(EOS_MODELS))
        sp.add_argument("--T-unit", default="K", choices=["K", "C", "F"])
        sp.add_argument("--P-unit", default="bar", choices=list(_P_TO_BAR))
        sp.add_argument("--V-unit", default="cm3/mol", choices=list(_V_TO_CM3))

    sub.add_parser("list-species", help="list built-in species")
    sub.add_parser("eos-table", help="print the parameter table of the four equations of state")

    z = sub.add_parser("z", help="state at given T and P")
    common(z); z.add_argument("--T", type=float, required=True); z.add_argument("--P", type=float, required=True)
    z.add_argument("--all-eos", action="store_true", help="run all four EOS")

    pp = sub.add_parser("p", help="pressure at given T and V")
    common(pp); pp.add_argument("--T", type=float, required=True); pp.add_argument("--V", type=float, required=True)
    pp.add_argument("--all-eos", action="store_true")

    t = sub.add_parser("t", help="temperature at given P and V")
    common(t); t.add_argument("--P", type=float, required=True); t.add_argument("--V", type=float, required=True)
    t.add_argument("--all-eos", action="store_true")

    s = sub.add_parser("sat", help="saturation pressure and volumes at T < Tc")
    common(s); s.add_argument("--T", type=float, required=True)
    s.add_argument("--all-eos", action="store_true")

    pr = sub.add_parser("props", help="H and S (residual + total) at T, P")
    common(pr); pr.add_argument("--T", type=float, required=True); pr.add_argument("--P", type=float, required=True)
    pr.add_argument("--all-eos", action="store_true")
    pr.add_argument("--phase", choices=["liquid", "vapor"], help="force a root (default: stable phase)")

    dl = sub.add_parser("delta", help="dH and dS between two states (may cross phase boundary)")
    common(dl)
    for k in ("T1", "P1", "T2", "P2"):
        dl.add_argument("--" + k, type=float, required=True)
    dl.add_argument("--phase1", choices=["liquid", "vapor"]); dl.add_argument("--phase2", choices=["liquid", "vapor"])
    dl.add_argument("--all-eos", action="store_true")

    hv = sub.add_parser("hvap", help="latent heat and entropy of vaporization, Psat, at T < Tc")
    common(hv); hv.add_argument("--T", type=float, required=True); hv.add_argument("--all-eos", action="store_true")

    iso = sub.add_parser("isotherm", help="tabulate P(V) or Z(P) along an isotherm")
    common(iso); iso.add_argument("--T", type=float, required=True)
    iso.add_argument("--Pmax", type=float, required=True); iso.add_argument("--n", type=int, default=15)

    c = sub.add_parser("compare", help="all EOS + ideal gas + virial at one (T, P)")
    common(c, need_eos=False); c.add_argument("--T", type=float, required=True); c.add_argument("--P", type=float, required=True)

    pl = sub.add_parser("plot", help="PV isotherm or Z-P plot (needs matplotlib)")
    common(pl); pl.add_argument("--kind", choices=["pv", "z"], default="pv")
    pl.add_argument("--T", type=float, help="needed for kind=z")
    pl.add_argument("--Pmax", type=float, help="needed for kind=z")
    pl.add_argument("--out", default=None)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "list-species":
        print("\n".join(list_species()))
        return 0
    if a.cmd == "eos-table":
        print("P = RT/(V-b) - a(T)/(V^2 + u b V + w b^2),  b = Omega R Tc/Pc,  a(Tc) = Psi R^2 Tc^2/Pc\n")
        print(f"{'EOS':<5}{'u':>6}{'w':>6}{'sigma':>8}{'eps':>8}{'Omega':>9}{'Psi':>9}{'Zc':>8}   a(T)")
        atxt = {"vdW": "constant", "RK": "Tr^-1/2", "SRK": "f(Tr, omega)", "PR": "f(Tr, omega)"}
        for n, e in EOS_MODELS.items():
            print(f"{n:<5}{e.u:6.3g}{e.w:6.3g}{e.sigma:8.4f}{e.eps:8.4f}{e.Omega:9.5f}{e.Psi:9.5f}{e.Zc:8.4f}   {atxt[n]}")
        return 0
    try:
        sp = _species_from_args(a)
    except KeyError as e:
        print(e, file=sys.stderr)
        return 2
    eos_list = list(EOS_MODELS) if getattr(a, "all_eos", False) else [getattr(a, "eos", "PR")]
    print(f"Species: {sp.name}   Tc = {sp.Tc} K, Pc = {sp.Pc} bar, omega = {sp.omega}")

    try:
        if a.cmd == "z":
            T, P = T_to_K(a.T, a.T_unit), P_to_bar(a.P, a.P_unit)
            for e in eos_list:
                m = CubicEOS(sp, e); _set_ref(m, a)
                _print_state(m.state(T, P), a)

        elif a.cmd == "props":
            T, P = T_to_K(a.T, a.T_unit), P_to_bar(a.P, a.P_unit)
            for e in eos_list:
                m = CubicEOS(sp, e); _set_ref(m, a)
                _print_state(m.state(T, P, a.phase), a)

        elif a.cmd == "delta":
            T1, T2 = T_to_K(a.T1, a.T_unit), T_to_K(a.T2, a.T_unit)
            P1, P2 = P_to_bar(a.P1, a.P_unit), P_to_bar(a.P2, a.P_unit)
            for e in eos_list:
                m = CubicEOS(sp, e); _set_ref(m, a)
                r = m.delta(T1, P1, T2, P2, a.phase1, a.phase2)
                s1, s2 = r["state1"], r["state2"]
                print(f"\n{e}: state 1 = {s1.phase}, Z = {s1.Z:.5f};  state 2 = {s2.phase}, Z = {s2.Z:.5f}")
                print(f"  H^R1 = {s1.H_res:.6g}, H^R2 = {s2.H_res:.6g} J/mol;  S^R1 = {s1.S_res:.6g}, S^R2 = {s2.S_res:.6g} J/(mol K)")
                print(f"  dH = {r['dH']:.6g} J/mol  ({r['dH'] / 1000:.5g} kJ/mol)")
                print(f"  dS = {r['dS']:.6g} J/(mol K)")

        elif a.cmd == "hvap":
            T = T_to_K(a.T, a.T_unit)
            for e in eos_list:
                m = CubicEOS(sp, e); _set_ref(m, a)
                v = m.vaporization(T)
                print(f"\n{e}: Psat = {P_from_bar(v['Psat'], a.P_unit):.6g} {a.P_unit}")
                print(f"  dH_vap = {v['dH_vap']:.6g} J/mol ({v['dH_vap'] / 1000:.5g} kJ/mol)   dS_vap = {v['dS_vap']:.6g} J/(mol K)")
                print(f"  H^R liq = {v['HR_liq']:.6g}, vap = {v['HR_vap']:.6g} J/mol;  S^R liq = {v['SR_liq']:.6g}, vap = {v['SR_vap']:.6g} J/(mol K)")

        elif a.cmd == "p":
            T, V = T_to_K(a.T, a.T_unit), V_to_cm3(a.V, a.V_unit)
            for e in eos_list:
                m = CubicEOS(sp, e)
                P = m.pressure(T, V)
                print(f"{e:>4}: P = {P_from_bar(P, a.P_unit):.6g} {a.P_unit}   "
                      f"Z = {P * V / (R_GAS * T):.5f}   (ideal gas: {P_from_bar(R_GAS * T / V, a.P_unit):.6g} {a.P_unit})")

        elif a.cmd == "t":
            P, V = P_to_bar(a.P, a.P_unit), V_to_cm3(a.V, a.V_unit)
            for e in eos_list:
                T = CubicEOS(sp, e).temperature(P, V)
                print(f"{e:>4}: T = {T_from_K(T, a.T_unit):.5g} {a.T_unit.upper()}  ({T:.3f} K)")

        elif a.cmd == "sat":
            T = T_to_K(a.T, a.T_unit)
            for e in eos_list:
                Ps, Vl, Vv = CubicEOS(sp, e).saturation(T)
                print(f"{e:>4}: Psat = {P_from_bar(Ps, a.P_unit):.6g} {a.P_unit}   "
                      f"V_liq = {V_from_cm3(Vl, a.V_unit):.5g}   V_vap = {V_from_cm3(Vv, a.V_unit):.5g} {a.V_unit}")

        elif a.cmd == "isotherm":
            T = T_to_K(a.T, a.T_unit); Pmax = P_to_bar(a.Pmax, a.P_unit)
            m = CubicEOS(sp, a.eos)
            print(f"{'P':>12} {'Z':>9} {'V':>14}  phase")
            for P in np.linspace(Pmax / a.n, Pmax, a.n):
                s = m.state(T, P)
                print(f"{P_from_bar(P, a.P_unit):12.5g} {s.Z:9.5f} {V_from_cm3(s.V, a.V_unit):14.6g}  {s.phase}")

        elif a.cmd == "compare":
            T, P = T_to_K(a.T, a.T_unit), P_to_bar(a.P, a.P_unit)
            print(f"T = {T:.2f} K, P = {P:.4g} bar, Tr = {T / sp.Tc:.3f}, Pr = {P / sp.Pc:.3f}\n")
            print(f"{'model':<18}{'Z':>9}{'V [cm3/mol]':>14}  phase")
            print(f"{'ideal gas':<18}{1.0:9.5f}{R_GAS * T / P:14.5g}")
            Zv = virial_pitzer_Z(sp, T, P)
            print(f"{'Pitzer virial B':<18}{Zv:9.5f}{Zv * R_GAS * T / P:14.5g}  (valid only at low/moderate Pr)")
            for e in EOS_MODELS:
                s = CubicEOS(sp, e).state(T, P)
                print(f"{e:<18}{s.Z:9.5f}{s.V:14.5g}  {s.phase}")

        elif a.cmd == "plot":
            if a.kind == "pv":
                out = a.out or f"{sp.name}_{a.eos}_PV.png"
                print("saved", plot_pv(sp, a.eos, out=out))
            else:
                if a.T is None or a.Pmax is None:
                    print("--T and --Pmax are required for kind=z", file=sys.stderr)
                    return 2
                out = a.out or f"{sp.name}_Z.png"
                print("saved", plot_z(sp, T_to_K(a.T, a.T_unit), P_to_bar(a.Pmax, a.P_unit), out))
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
