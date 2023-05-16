#!/usr/bin/env python
""":mod:`primpy.units`: units and constants for primpy."""
import numpy as np
from scipy.constants import mega, giga, parsec as parsec_m
from scipy.constants import pi, c, hbar, G, k as k_B, e

# reduced Planck units in SI
mp_kg = np.sqrt(hbar * c / (8 * pi * G))
tp_s = np.sqrt(8 * pi * G * hbar / c**5)
lp_m = np.sqrt(8 * pi * G * hbar / c**3)
Tp_K = np.sqrt(hbar * c**5 / (8 * pi * G * k_B**2))

# reduced Planck units in GeV
mp_GeV = mp_kg * c**2 / (giga * e)
tp_iGeV = tp_s / hbar * giga * e
lp_iGeV = lp_m / hbar / c * giga * e

# other units
Mpc_m = mega * parsec_m

# derived constants
a_B = 8 * pi**5 * k_B**4 / 15 / (2 * pi * hbar)**3 / c**3  # radiation constant (Planck's law)
