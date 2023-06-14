#!/usr/bin/env python
"""Tests for `primpy.time.inflation` module."""
import pytest
import numpy as np
from primpy.potentials import QuadraticPotential
from primpy.time.inflation import InflationEquationsT


@pytest.mark.parametrize('K', [-1, 0, +1])
def test_basic_methods(K):
    eq = InflationEquationsT(K=K, potential=QuadraticPotential(Lambda=1))
    assert hasattr(eq, 'phi')
    assert hasattr(eq, 'dphidt')
    y0 = np.zeros(len(eq.idx))
    assert eq.H2(x=0, y=y0) == -K
    y1 = np.ones(len(eq.idx))
    V = 1
    H2 = (1 / 2 + V) / 3 - K * np.exp(-2)
    assert eq.H2(x=1, y=y1) == H2
    assert eq.H(x=1, y=y1) == np.sqrt(H2)
    assert eq.w(x=1, y=y1) == (1/2 - V) / (1/2 + V)
    assert eq.inflating(x=1, y=y1) == V - 1
