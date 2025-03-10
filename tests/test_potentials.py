#!/usr/bin/env python
"""Tests for `primpy.potential` module."""
import pytest
from pytest import approx
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from primpy.exceptionhandling import PrimpyError, PrimpyWarning
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
                                             (pp.DoubleWell4Potential, dict(phi0=100)),
                                             (pp.TmodelPotential, dict(p=2, alpha=1))])
@pytest.mark.parametrize('Lambda, phi', [(1, 3.), (2e-3, 10.)])
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
    pot.d4V(phi=phi)
    assert pot.inv_V(V=Lambda**4/2) > 0
    assert 0 < pot.phi_end < phi
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
    e1V = pot.get_epsilon_1V(phi=phi)
    assert 0 < e1V < 1
    pot.get_epsilon_2V(phi=phi)
    pot.get_epsilon_3V(phi=phi)
    pot.get_epsilon_4V(phi=phi)
    e1 = pot.get_epsilon_1(phi=phi)
    assert 0 < e1 < 1
    pot.get_epsilon_2(phi=phi)
    pot.get_epsilon_3(phi=phi)
    pot.get_epsilon_4(phi=phi)
    Pot(A_s=2e-9, N_star=60, **pot_kwargs)
    pot2 = Pot(A_s=2e-9, phi_star=5, **pot_kwargs)
    assert pot2.Lambda == approx(L)
    with pytest.warns(PrimpyWarning):
        Pot(A_s=2e-9, N_star=60, Lambda=1e-2, **pot_kwargs)
    with pytest.raises(PrimpyError):
        Pot(A_s=2e-9, **pot_kwargs)


@pytest.mark.parametrize('Lambda, phi', [(1, 1), (0.0025, 20)])
def test_quadratic_inflation_V(Lambda, phi):
    """Tests for `QuadraticPotential`."""
    pot1 = pp.QuadraticPotential(Lambda=Lambda)
    assert pot1.V(phi=phi) == approx(Lambda**4 * phi**2)
    assert pot1.dV(phi=phi) == approx(2 * Lambda**4 * phi)
    assert pot1.d2V(phi=phi) == approx(2 * Lambda**4)
    assert pot1.d3V(phi=phi) == approx(0)
    assert pot1.inv_V(V=Lambda**4) == approx(np.sqrt(1))
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

    with pytest.warns(DeprecationWarning):
        Pot.phi2efolds(phi0/2, phi0=phi0)


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
    assert Pot.sr_ns2Nstar(n_s=n_s, p=p) == approx(N_star)

    r = Pot.sr_Nstar2r(N_star=N_star, p=p)
    assert 1e-2 < Pot.sr_Nstar2r(N_star=N_star, p=p) < 1
    assert r == 16 * p / (4 * N_star + p)
    assert Pot.sr_r2Nstar(r=r, p=p) == approx(N_star)


@pytest.mark.parametrize('Pot, p', [(pp.LinearPotential, 1),
                                    (pp.QuadraticPotential, 2),
                                    (pp.CubicPotential, 3),
                                    (pp.QuarticPotential, 4)])
@pytest.mark.parametrize('N_star', [20, 60, 90])
def test_specific_monomial_slow_roll(Pot, p, N_star):
    n_s = Pot.sr_Nstar2ns(N_star=N_star)
    assert 0.8 < n_s < 1
    assert n_s == 1 - p / (2 * N_star) - 1 / N_star
    assert Pot.sr_ns2Nstar(n_s=n_s) == approx(N_star)

    r = Pot.sr_Nstar2r(N_star=N_star)
    assert 1e-2 < Pot.sr_Nstar2r(N_star=N_star) < 1
    assert r == 16 * p / (4 * N_star + p)
    assert Pot.sr_r2Nstar(r=r) == approx(N_star)


@pytest.mark.parametrize('N_star', [20, 60, 90])
def test_starobinsky_slow_roll(N_star):
    Pot = pp.StarobinskyPotential

    n_s = Pot.sr_Nstar2ns(N_star=N_star)
    aprx = 1 - 2 / N_star + (-3 + np.sqrt(3)) / N_star**2 + (-3 + 3*np.sqrt(3)) / N_star**3
    assert 0.8 < n_s < 1
    assert n_s == approx(aprx, rel=1e-3)
    assert Pot.sr_ns2Nstar(n_s=n_s) == approx(N_star)

    r = Pot.sr_Nstar2r(N_star=N_star)
    aprx = 12 / N_star**2 - 12 * np.sqrt(3) / N_star**3 + 27 / N_star**4
    assert 1e-3 < r < 1
    assert r == approx(aprx, rel=1e-3)
    assert Pot.sr_r2Nstar(r=r) == approx(N_star)

    with pytest.warns(DeprecationWarning):
        Pot.phi2efolds(phi=5)


@pytest.mark.parametrize('pot_kwargs', [dict(phi0=10), dict(phi0=100), dict(phi0=1000)])
@pytest.mark.parametrize('N_star', [20, 60, 90])
def test_natural_slow_roll(pot_kwargs, N_star):
    Pot = pp.NaturalPotential
    n_s = Pot.sr_Nstar2ns(N_star=N_star, **pot_kwargs)
    assert 0.8 < n_s < 1
    assert Pot.sr_ns2Nstar(n_s=n_s, **pot_kwargs) == approx(N_star)
    with pytest.raises(PrimpyError):
        Pot.sr_ns2Nstar(n_s=n_s, phi0=1)

    r = Pot.sr_Nstar2r(N_star=N_star, **pot_kwargs)
    assert 1e-4 < r < 1
    assert Pot.sr_r2Nstar(r=r, **pot_kwargs) == approx(N_star)

    with pytest.warns(DeprecationWarning):
        Pot.phi2efolds(phi=pot_kwargs['phi0']/2, phi0=pot_kwargs['phi0'])


@pytest.mark.parametrize('a_feature', [0.1, 0.01])
@pytest.mark.parametrize('b_feature', [0.5, 0.05])
def test_feature_potential(a_feature, b_feature):
    eps = np.finfo(float).eps
    Lambda = 1e-3
    phi_feature = 5
    phi = np.linspace(0, 10, 200 + 1)[1:]
    phi_smaller = phi[phi < phi_feature]
    phi_greater = phi[phi > phi_feature]
    phi_outside = phi[np.argwhere((phi < phi_feature - 3 * b_feature) |
                                  (phi > phi_feature + 4 * b_feature)).ravel()]
    pot = pp.StarobinskyPotential(Lambda=Lambda)

    # Gaussian dip
    fpot = pp.StarobinskyGaussianDipPotential(Lambda=Lambda, phi_feature=phi_feature,
                                              a_feature=a_feature, b_feature=b_feature)
    max_dV = np.max(np.abs(fpot.dV(phi=phi) / pot.dV(phi=phi) - 1))
    max_d2V = np.max(np.abs(fpot.d2V(phi=phi) / pot.d2V(phi=phi) - 1))
    max_d3V = np.max(np.abs(fpot.d3V(phi=phi) / pot.d3V(phi=phi) - 1))
    assert np.all(fpot.V(phi=phi) >= 0)
    assert np.all(fpot.V(phi=phi) <= pot.V(phi=phi))
    assert_allclose(fpot.V(phi=phi) / pot.V(phi=phi), 1, rtol=a_feature + eps)
    assert_allclose(fpot.dV(phi_outside) / pot.dV(phi_outside), 1, rtol=max_dV/10)
    assert_allclose(fpot.d2V(phi_outside) / pot.d2V(phi_outside), 1, rtol=max_d2V/10)
    assert_allclose(fpot.d3V(phi_outside) / pot.d3V(phi_outside), 1, rtol=max_d3V/10)
    assert np.mean(fpot.dV(phi=phi_smaller) / pot.dV(phi=phi_smaller) - 1) < 0
    assert np.mean(fpot.dV(phi=phi_greater) / pot.dV(phi=phi_greater) - 1) > 0
    assert np.mean(fpot.d3V(phi=phi_smaller) / pot.d3V(phi=phi_smaller) - 1) > 0
    assert np.mean(fpot.d3V(phi=phi_greater) / pot.d3V(phi=phi_greater) - 1) < 0

    # Tanh step
    fpot = pp.StarobinskyTanhStepPotential(Lambda=Lambda, phi_feature=phi_feature,
                                           a_feature=a_feature, b_feature=b_feature)
    max_dV = np.max(np.abs(fpot.dV(phi=phi) / pot.dV(phi=phi) - 1))
    max_d2V = np.max(np.abs(fpot.d2V(phi=phi) / pot.d2V(phi=phi) - 1))
    max_d3V = np.max(np.abs(fpot.d3V(phi=phi) / pot.d3V(phi=phi) - 1))
    assert np.all(fpot.V(phi=phi) >= 0)
    assert np.all(fpot.V(phi=phi_smaller) <= pot.V(phi=phi_smaller))
    assert np.all(fpot.V(phi=phi_greater) >= pot.V(phi=phi_greater))
    assert_allclose(fpot.V(phi=phi) / pot.V(phi=phi), 1, rtol=a_feature + eps)
    assert_allclose(fpot.dV(phi_outside) / pot.dV(phi_outside), 1, rtol=max_dV/10)
    assert_allclose(fpot.d2V(phi_outside) / pot.d2V(phi_outside), 1, rtol=max_d2V/10)
    assert_allclose(fpot.d3V(phi_outside) / pot.d3V(phi_outside), 1, rtol=max_d3V/10)
    assert np.mean(fpot.dV(phi=phi) / pot.dV(phi=phi) - 1) > 0
    assert np.mean(fpot.d2V(phi=phi_smaller) / pot.d2V(phi=phi_smaller) - 1) < 0
    assert np.mean(fpot.d2V(phi=phi_greater) / pot.d2V(phi=phi_greater) - 1) > 0
