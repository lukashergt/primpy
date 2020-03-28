#!/usr/bin/env python
"""Tests for `primpy.time.inflation` module."""
import numpy as np
from primpy.potentials import QuadraticPotential
from primpy.time.inflation import InflationEquationsT


def test_basic_methods():
    for K in [-1, 0, 1]:
        eq = InflationEquationsT(K=K, potential=QuadraticPotential(mass=1))
        assert eq.H2(x=0, y=np.zeros(4)) == -K
        assert eq.H2(x=1, y=np.ones(4)) == 1 / 3 - K * np.exp(-2)
        assert eq.H(x=1, y=np.ones(4)) == np.sqrt(1 / 3 - K * np.exp(-2))
        assert eq.w(x=1, y=np.ones(4)) == 0
        assert eq.inflating(x=1, y=np.ones(4)) == -0.5
