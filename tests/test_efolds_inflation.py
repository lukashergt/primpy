#!/usr/bin/env python
"""Tests for `primpy.efolds.inflation` module."""
import pytest
import numpy as np
from primpy.potentials import QuadraticPotential
from primpy.efolds.inflation import InflationEquationsN


@pytest.mark.parametrize('K', [-1, 0, +1])
def test_basic_methods(K):
    eq = InflationEquationsN(K=K, potential=QuadraticPotential(Lambda=1))
    assert hasattr(eq, 'phi')
    assert hasattr(eq, 'dphidN')
    y0 = np.zeros(len(eq.idx))
    assert eq.H2(x=0, y=y0) == -K
    y1 = np.ones(len(eq.idx))
    V = 1
    H2 = (2 * V - 6 * K * np.exp(-2)) / 5
    assert eq.H2(x=1, y=y1) == H2
    assert eq.H(x=1, y=y1) == np.sqrt(H2)
    assert eq.w(x=1, y=y1) == (H2/2 - V) / (H2/2 + V)
    assert eq.inflating(x=1, y=y1) == V - H2
