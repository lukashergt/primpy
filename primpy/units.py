"""Units and constants for primpy."""
import numpy as np
from scipy.constants import mega, giga, parsec as parsec_m
from scipy.constants import pi, c, hbar, G, k as k_B, e

# reduced Planck units in SI
mp_kg = np.sqrt(hbar * c / (8 * pi * G))             #: reduced Planck mass in kg
tp_s = np.sqrt(8 * pi * G * hbar / c**5)             #: reduced Planck time in s
lp_m = np.sqrt(8 * pi * G * hbar / c**3)             #: reduced Planck length in m
Tp_K = np.sqrt(hbar * c**5 / (8 * pi * G * k_B**2))  #: reduced Planck temperature in K

# reduced Planck units in GeV
mp_GeV = mp_kg * c**2 / (giga * e)    #: reduced Planck mass in GeV
tp_iGeV = tp_s / hbar * giga * e      #: reduced Planck time in GeV^-1
lp_iGeV = lp_m / hbar / c * giga * e  #: reduced Planck length in GeV^-1

# other units
Mpc_m = mega * parsec_m  #: conversion factor from Mpc to m

# derived constants
a_B = 8 * pi**5 * k_B**4 / 15 / (2 * pi * hbar)**3 / c**3  #: radiation constant (Planck's law)
