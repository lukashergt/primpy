#!/usr/bin/env python
"""Tests for `primpy.potential` module."""
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
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


@pytest.mark.parametrize('Lambda, phi', [(1, 1), (0.0025, 20)])
def test_quadratic_inflation_V(Lambda, phi):
    """Tests for `QuadraticPotential`."""
    pot1 = pp.QuadraticPotential(Lambda=Lambda)
    assert pot1.V(phi=phi) == pytest.approx(Lambda**4 * phi**2)
    assert pot1.dV(phi=phi) == pytest.approx(2 * Lambda**4 * phi)
    assert pot1.d2V(phi=phi) == pytest.approx(2 * Lambda**4)
    assert pot1.d3V(phi=phi) == pytest.approx(0)
    assert pot1.inv_V(V=Lambda**4) == pytest.approx(np.sqrt(1))
    with pytest.raises(Exception):
        pp.QuadraticPotential(mass=Lambda**2)


def test_quadratic_inflation_power_to_potential():
    pot = pp.QuadraticPotential(Lambda=0.0025)
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


@pytest.mark.parametrize('Pot', [pp.DoubleWell2Potential,
                                 pp.DoubleWell4Potential])
@pytest.mark.parametrize('phi0', np.logspace(1, 3, 10))
def test_doublewell_inflation_V(Pot, phi0):
    """Tests for `StarobinskyPotential`."""
    phi = np.linspace(5, 9, 5)
    Lambda = 1e-3
    pot = Pot(Lambda=Lambda, phi0=phi0)

    pot.V(phi=phi)
    pot.dV(phi=phi)
    pot.d2V(phi=phi)
    pot.d3V(phi=phi)
    assert_array_equal(phi, np.linspace(5, 9, 5))

    assert_allclose(
        pot.V(phi=phi),
        Lambda**4 * (-1 + (-1 + phi / phi0)**pot.p)**2,
        rtol=1e-12, atol=1e-12)
    assert_allclose(
        pot.dV(phi=phi),
        (2 * pot.p * Lambda**4 * (-1 + phi / phi0)**pot.p *
         (-1 + (-1 + phi / phi0)**pot.p)) / (phi0 - phi),
        rtol=1e-12, atol=1e-12)
    assert_allclose(
        pot.d2V(phi=phi),
        (2 * pot.p * Lambda**4 * (-1 + phi / phi0)**pot.p *
         (1 - pot.p + (-1 + 2 * pot.p) * (-1 + phi / phi0)**pot.p)) / (phi0 - phi)**2,
        rtol=1e-12, atol=1e-12)
    assert_allclose(
        pot.d3V(phi=phi),
        (2 * (-1 + pot.p) * pot.p * Lambda**4 * (-1 + phi / phi0)**pot.p *
         (2 - pot.p + 2 * (-1 + 2 * pot.p) * (-1 + phi / phi0)**pot.p)) / (phi0 - phi)**3,
        rtol=1e-12, atol=1e-12)


def test_starobinsky_inflation_power_to_potential():
    pot = pp.StarobinskyPotential(Lambda=1e-3)
    assert 0 < pot.sr_As2Lambda(2e-9, None, 55)[1] < 10
    assert 0 < pot.sr_As2Lambda(2e-9, 5, None)[2] < 100


@pytest.mark.parametrize('p', [2/3, 1, 4/3, 2, 4])
@pytest.mark.parametrize('N_star', [20, 60, 90])
def test_monomial_slow_roll(p, N_star):
    Pot = pp.MonomialPotential
    n_s = Pot.sr_Nstar2ns(N_star=N_star, p=p)
    assert 0.8 < n_s < 1
    assert n_s == 1 - p / (2 * N_star) - 1 / N_star
    assert Pot.sr_ns2Nstar(n_s=n_s, p=p) == pytest.approx(N_star)

    r = Pot.sr_Nstar2r(N_star=N_star, p=p)
    assert 1e-2 < Pot.sr_Nstar2r(N_star=N_star, p=p) < 1
    assert r == 16 * p / (4 * N_star + p)
    assert Pot.sr_r2Nstar(r=r, p=p) == pytest.approx(N_star)


@pytest.mark.parametrize('Pot, p', [(pp.LinearPotential, 1),
                                    (pp.QuadraticPotential, 2),
                                    (pp.CubicPotential, 3),
                                    (pp.QuarticPotential, 4)])
@pytest.mark.parametrize('N_star', [20, 60, 90])
def test_specific_monomial_slow_roll(Pot, p, N_star):
    n_s = Pot.sr_Nstar2ns(N_star=N_star)
    assert 0.8 < n_s < 1
    assert n_s == 1 - p / (2 * N_star) - 1 / N_star
    assert Pot.sr_ns2Nstar(n_s=n_s) == pytest.approx(N_star)

    r = Pot.sr_Nstar2r(N_star=N_star)
    assert 1e-2 < Pot.sr_Nstar2r(N_star=N_star) < 1
    assert r == 16 * p / (4 * N_star + p)
    assert Pot.sr_r2Nstar(r=r) == pytest.approx(N_star)


@pytest.mark.parametrize('N_star', [20, 60, 90])
def test_starobinsky_slow_roll(N_star):
    Pot = pp.StarobinskyPotential

    n_s = Pot.sr_Nstar2ns(N_star=N_star)
    approx = 1 - 2 / N_star + (-3 + np.sqrt(3)) / N_star**2 + (-3 + 3*np.sqrt(3)) / N_star**3
    assert 0.8 < n_s < 1
    assert n_s == pytest.approx(approx, rel=1e-3)
    assert Pot.sr_ns2Nstar(n_s=n_s) == pytest.approx(N_star)

    r = Pot.sr_Nstar2r(N_star=N_star)
    approx = 12 / N_star**2 - 12 * np.sqrt(3) / N_star**3 + 27 / N_star**4
    assert 1e-3 < r < 1
    assert r == pytest.approx(approx, rel=1e-3)
    assert Pot.sr_r2Nstar(r=r) == pytest.approx(N_star)


@pytest.mark.parametrize('pot_kwargs', [dict(phi0=10), dict(phi0=100), dict(phi0=1000)])
@pytest.mark.parametrize('N_star', [20, 60, 90])
def test_natural_slow_roll(pot_kwargs, N_star):
    Pot = pp.NaturalPotential
    n_s = Pot.sr_Nstar2ns(N_star=N_star, **pot_kwargs)
    assert 0.8 < n_s < 1
    assert Pot.sr_ns2Nstar(n_s=n_s, **pot_kwargs) == pytest.approx(N_star)

    r = Pot.sr_Nstar2r(N_star=N_star, **pot_kwargs)
    assert 1e-4 < r < 1
    assert Pot.sr_r2Nstar(r=r, **pot_kwargs) == pytest.approx(N_star)
