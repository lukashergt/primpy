#!/usr/bin/env python
"""Tests for `primpy.potential` module."""
import pytest
from tests.test_tools import effequal
import numpy as np
import primpy.potentials as pp


@pytest.mark.parametrize('Pot, pot_kwargs', [(pp.MonomialPotential, dict(p=2/3)),
                                             (pp.LinearPotential, {}),
                                             (pp.QuadraticPotential, {}),
                                             (pp.CubicPotential, {}),
                                             (pp.QuarticPotential, {}),
                                             (pp.StarobinskyPotential, {}),
                                             (pp.NaturalPotential, dict(phi0=100)),
                                             (pp.DoubleWellPotential, dict(phi0=100, p=2)),
                                             (pp.DoubleWell2Potential, dict(phi0=100)),
                                             (pp.DoubleWell4Potential, dict(phi0=100))])
@pytest.mark.parametrize('Lambda, phi', [(1, 1), (2e-3, 10)])
def test_inflationary_potentials(Pot, pot_kwargs, Lambda, phi):
    with pytest.raises(Exception):
        kwargs = pot_kwargs.copy()
        kwargs['foo'] = 0
        Pot(Lambda=Lambda, **kwargs)
    pot = Pot(Lambda=Lambda, **pot_kwargs)
    assert isinstance(pot.tag, str)
    assert isinstance(pot.name, str)
    assert isinstance(pot.tex, str)
    assert pot.V(phi=phi) > 0
    assert pot.dV(phi=phi) > 0
    pot.d2V(phi=phi)
    pot.d3V(phi=phi)
    assert pot.inv_V(V=Lambda**4/2) > 0
    if type(pot) == pp.DoubleWellPotential:
        with pytest.raises(NotImplementedError):
            pot.sr_As2Lambda(A_s=2e-9, phi_star=None, N_star=60, **pot_kwargs)
    else:
        L, p, N = pot.sr_As2Lambda(A_s=2e-9, phi_star=None, N_star=60, **pot_kwargs)
        assert L > 0
        assert p > 0
        assert N == 60
        L, p, N = pot.sr_As2Lambda(A_s=2e-9, phi_star=5, N_star=None, **pot_kwargs)
        assert L > 0
        assert p == 5
        assert 0 < N < 100
        with pytest.raises(Exception):
            pot.sr_As2Lambda(A_s=2e-9, phi_star=5, N_star=60, **pot_kwargs)


@pytest.mark.parametrize('mass, phi', [(1, 1), (6e-6, 20)])
def test_quadratic_inflation_V(mass, phi):
    """Tests for `QuadraticPotential`."""
    pot1 = pp.QuadraticPotential(Lambda=np.sqrt(mass))
    assert pot1.V(phi=phi) == effequal(0.5 * mass**2 * phi**2)
    assert pot1.dV(phi=phi) == effequal(mass**2 * phi)
    assert pot1.d2V(phi=phi) == effequal(mass**2)
    assert pot1.d3V(phi=phi) == effequal(0)
    assert pot1.inv_V(V=mass**2) == effequal(np.sqrt(2))
    pot2 = pp.QuadraticPotential(mass=mass)
    assert pot1.V(phi=phi) == pot2.V(phi=phi)
    with pytest.raises(Exception):
        pp.QuadraticPotential(mass=mass, Lambda=np.sqrt(mass))


def test_quadratic_inflation_power_to_potential():
    pot = pp.QuadraticPotential(Lambda=np.sqrt(6e-6))
    assert pot.sr_As2Lambda(2e-9, None, 55)[1] == np.sqrt(4 * 55 + 2)
    assert pot.sr_As2Lambda(2e-9, 20, None)[2] == (20 ** 2 - 2) / 4


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
    assert pot.inv_V(V=Lambda**4/2) == -np.log(1 - np.sqrt(1/2)) / gamma


def test_starobinsky_inflation_power_to_potential():
    pot = pp.StarobinskyPotential(Lambda=1e-3)
    assert 0 < pot.sr_As2Lambda(2e-9, None, 55)[1] < 10
    assert 0 < pot.sr_As2Lambda(2e-9, 5, None)[2] < 100


@pytest.mark.parametrize('Pot, pot_kwargs', [(pp.NaturalPotential, dict(phi0=20)),
                                             (pp.NaturalPotential, dict(phi0=50)),
                                             (pp.NaturalPotential, dict(phi0=100)),
                                             (pp.NaturalPotential, dict(phi0=200)),
                                             (pp.NaturalPotential, dict(phi0=500))])
@pytest.mark.parametrize('N_star', [50, 60])
def test_slow_roll_methods(Pot, pot_kwargs, N_star, ):
    assert 0.9 < Pot.sr_n_s(N_star=N_star, **pot_kwargs) < 1
    assert 1e-3 < Pot.sr_r(N_star=N_star, **pot_kwargs) < 1
