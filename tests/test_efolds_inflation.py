#!/usr/bin/env python
"""Tests for `primpy.efolds.inflation` module."""
import numpy as np
from primpy.potentials import QuadraticPotential
from primpy.efolds.inflation import InflationEquationsN
from primpy.initialconditions import InflationStartIC_NiPi


def test_basic_methods():
    for K in [-1, 0, 1]:
        eq = InflationEquationsN(K=K, potential=QuadraticPotential(mass=1))
        assert hasattr(eq, 'phi')
        assert hasattr(eq, 'dphidN')
        y0 = np.zeros(len(eq.idx))
        assert eq.H2(x=0, y=y0) == -K
        y1 = np.ones(len(eq.idx))
        H2 = (1 - 6 * K * np.exp(-2)) / 5
        assert eq.H2(x=1, y=y1) == H2
        assert eq.H(x=1, y=y1) == np.sqrt(H2)
        assert eq.w(x=1, y=y1) == (H2 - 1) / (H2 + 1)
        assert eq.inflating(x=1, y=y1) == 1 / 2 - H2


def test_track_eta():
    pot = QuadraticPotential(mass=1)
    N_i = 10
    phi_i = 17
    eta_i = 0
    for K in [-1, 0, 1]:
        eq = InflationEquationsN(K=K, potential=pot, track_eta=True)
        assert eq.track_eta
        assert hasattr(eq, 'phi')
        assert hasattr(eq, 'dphidN')
        assert hasattr(eq, 'eta')
        assert 'eta' in eq.idx
        ic = InflationStartIC_NiPi(equations=eq, N_i=N_i, phi_i=phi_i, eta_i=eta_i)
        y0 = np.zeros(len(eq.idx))
        ic(y0=y0)
        dy0 = eq(x=ic.x_ini, y=y0)
        assert dy0.size == 3
        H2 = (2 * pot.V(phi_i) - 6 * K * np.exp(-2 * N_i)) / (6 - dy0[eq.idx['phi']]**2)
        assert dy0[eq.idx['eta']] == np.exp(-N_i) / np.sqrt(H2)
