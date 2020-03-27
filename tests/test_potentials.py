#!/usr/bin/env python
"""Tests for `primpy.potential` module."""
import numpy as np
from primpy.potentials import QuadraticPotential, StarobinskyPotential


def test_quadratic_inflation():
    """Tests for `QuadraticPotential`."""
    pot = QuadraticPotential(mass=1)
    assert pot.V(phi=1) == 0.5
    assert pot.dV(phi=1) == 1
    assert pot.d2V(phi=1) == 1
    assert pot.d3V(phi=1) == 0
    pot = QuadraticPotential(mass=6e-6)
    assert pot.V(phi=20) == 0.5 * 6e-6**2 * 20**2
    assert pot.As2mass(2e-9, 55)[1] == np.sqrt(4 * 55 + 2)


def test_starobinsky_inflation():
    """Tests for `StarobinskyPotential`."""
    gamma = np.sqrt(2 / 3)
    pot = StarobinskyPotential(Lambda=1)
    assert pot.V(phi=1) == (1 - np.exp(-gamma))**2
    assert pot.dV(phi=1) == 2 * gamma * np.exp(-2 * gamma) * (np.exp(gamma) - 1)
    assert pot.d2V(phi=1) == 2 * gamma**2 * np.exp(-2 * gamma) * (2 - np.exp(gamma))
    assert pot.d3V(phi=1) == 2 * gamma**3 * np.exp(-2 * gamma) * (np.exp(gamma) - 4)
    pot = StarobinskyPotential(Lambda=1e-2)
    assert pot.V(phi=15) == 1e-2**4 * (1 - np.exp(-np.sqrt(2 / 3) * 15))**2
