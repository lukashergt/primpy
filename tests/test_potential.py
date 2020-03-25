#!/usr/bin/env python
"""Tests for `primpy.potential` module."""
import numpy as np
from primpy.potential import QuadraticPotential, StarobinskyPotential


def test_quadratic_inflation():
    """Tests for `QuadraticPotential`."""
    pot = QuadraticPotential(m=1)
    assert pot.V(phi=1) == 0.5
    pot = QuadraticPotential(m=6e-6)
    assert pot.V(phi=20) == 0.5 * 6e-6**2 * 20**2


def test_starobinsky_inflation():
    """Tests for `StarobinskyPotential`."""
    pot = StarobinskyPotential(Lambda=1)
    assert pot.V(phi=1) == (1 - np.exp(-np.sqrt(2 / 3)))**2
    pot = StarobinskyPotential(Lambda=1e-2)
    assert pot.V(phi=15) == 1e-2**4 * (1 - np.exp(-np.sqrt(2 / 3) * 15))**2
