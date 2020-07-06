#!/usr/bin/env python
"""Tests for `primpy.potential` module."""
import pytest
import numpy as np
import primpy.potentials as pp


@pytest.mark.parametrize('mass, phi', [(1, 1), (6e-6, 20)])
def test_quadratic_inflation_V(mass, phi):
    """Tests for `QuadraticPotential`."""
    pot = pp.QuadraticPotential(mass=mass)
    assert pot.V(phi=phi) == 0.5 * mass**2 * phi**2
    assert pot.dV(phi=phi) == mass**2 * phi
    assert pot.d2V(phi=phi) == mass**2
    assert pot.d3V(phi=phi) == 0


def test_quadratic_inflation_extras():
    pot = pp.QuadraticPotential(mass=6e-6)
    assert pot.power_to_potential(2e-9, None, 55)[1] == np.sqrt(4 * 55 + 2)
    assert pot.power_to_potential(2e-9, 20, None)[2] == (20**2 - 2) / 4
    with pytest.raises(Exception):
        pot.power_to_potential(2e-9, 20, 55)


@pytest.mark.parametrize('Lambda, phi', [(1, 1), (1e-3, 10)])
def test_starobinsky_inflation_V(Lambda, phi):
    """Tests for `StarobinskyPotential`."""
    gamma = pp.StarobinskyPotential.gamma
    g_p = gamma * phi
    pot = pp.StarobinskyPotential(Lambda=Lambda)
    assert pot.V(phi=phi) == Lambda**4 * (1 - np.exp(-g_p))**2
    assert pot.dV(phi=phi) == Lambda**4 * 2 * gamma * np.exp(-2 * g_p) * (np.exp(g_p) - 1)
    assert pot.d2V(phi=phi) == Lambda**4 * 2 * gamma**2 * np.exp(-2 * g_p) * (2 - np.exp(g_p))
    assert pot.d3V(phi=phi) == Lambda**4 * 2 * gamma**3 * np.exp(-2 * g_p) * (np.exp(g_p) - 4)


def test_starobinsky_inflation_extras():
    pot = pp.StarobinskyPotential(Lambda=1e-3)
    assert 0 < pot.power_to_potential(2e-9, None, 55)[1] < 10
    assert 0 < pot.power_to_potential(2e-9, 5, None)[2] < 100
    with pytest.raises(Exception):
        pot.power_to_potential(2e-9, 20, 55)
