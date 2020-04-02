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
