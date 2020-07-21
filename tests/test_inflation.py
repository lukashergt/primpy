#!/usr/bin/env python
"""Tests for `primpy.inflation` module."""
import pytest
from pytest import approx
from scipy.interpolate import interp1d
import numpy as np
from numpy.testing import assert_allclose
from primpy.exceptionhandling import InflationEndWarning
from primpy.units import Mpc_m, lp_m
from primpy.parameters import K_STAR
from primpy.potentials import QuadraticPotential
from primpy.events import InflationEvent, UntilNEvent
from primpy.inflation import InflationEquations
from primpy.time.inflation import InflationEquationsT
from primpy.efolds.inflation import InflationEquationsN
from primpy.initialconditions import InflationStartIC, ISIC_NsOk
from primpy.solver import solve


def test_not_implemented_errors():
    eq = InflationEquations(K=1, potential=QuadraticPotential(Lambda=np.sqrt(6e-6)))
    with pytest.raises(NotImplementedError, match="Equations must define H2 method."):
        eq.H(x=0, y=np.zeros(4))
    with pytest.raises(NotImplementedError, match="Equations must define H2 method."):
        eq.H2(x=0, y=np.zeros(4))
    with pytest.raises(NotImplementedError, match="Equations must define w method."):
        eq.w(x=0, y=np.zeros(4))
    with pytest.raises(NotImplementedError, match="Equations must define inflating method."):
        eq.inflating(x=0, y=np.zeros(4))


def test_track_eta():
    pot = QuadraticPotential(Lambda=1)
    N_i = 10
    phi_i = 17
    eta_i = 0
    for K in [-1, 0, 1]:
        for eq in [InflationEquationsT(K=K, potential=pot, track_eta=True),
                   InflationEquationsN(K=K, potential=pot, track_eta=True)]:
            assert eq.track_eta
            assert hasattr(eq, 'phi')
            assert hasattr(eq, 'N')
            assert hasattr(eq, 'eta')
            assert 'eta' in eq.idx
            ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, eta_i=eta_i)
            y0 = np.zeros(len(eq.idx))
            ic(y0=y0)
            dy0 = eq(x=ic.x_ini, y=y0)
            if isinstance(eq, InflationEquationsT):
                assert hasattr(eq, 'dphidt')
                assert dy0.size == 4
                assert dy0[eq.idx['eta']] == np.exp(-N_i)
            elif isinstance(eq, InflationEquationsN):
                assert hasattr(eq, 'dphidN')
                assert dy0.size == 3
                H2 = (2 * pot.V(phi_i) - 6 * K * np.exp(-2 * N_i)) / (6 - dy0[eq.idx['phi']]**2)
                assert dy0[eq.idx['eta']] == np.exp(-N_i) / np.sqrt(H2)


def test_basic_methods_time_vs_efolds():
    tol = 1e-12
    t = 1
    N = 10
    phi = 20
    for K in [-1, 0, 1]:
        for Lambda in [1, np.sqrt(6e-6)]:
            pot = QuadraticPotential(Lambda=Lambda)
            for dphidt_squared in [100 * pot.V(phi), 2 * pot.V(phi), pot.V(phi), pot.V(phi) / 100]:
                dphidt = -np.sqrt(dphidt_squared)
                eq_t = InflationEquationsT(K=K, potential=pot)
                eq_N = InflationEquationsN(K=K, potential=pot)
                assert eq_t.idx['phi'] == 0
                assert eq_t.idx['dphidt'] == 1
                assert eq_t.idx['N'] == 2
                assert eq_N.idx['phi'] == 0
                assert eq_N.idx['dphidN'] == 1
                y1_t = np.array([phi, dphidt, N])
                y1_N = np.array([phi, dphidt / eq_t.H(t, y1_t)])
                assert eq_t.a(t, y1_t) == approx(eq_N.a(N, y1_N), rel=tol, abs=tol)
                assert eq_t.H2(t, y1_t) == approx(eq_N.H2(N, y1_N), rel=tol, abs=tol)
                assert eq_t.H(t, y1_t) == approx(eq_N.H(N, y1_N), rel=tol, abs=tol)
                assert eq_t.V(t, y1_t) == approx(eq_N.V(N, y1_N), rel=tol, abs=tol)
                assert eq_t.dVdphi(t, y1_t) == approx(eq_N.dVdphi(N, y1_N), rel=tol, abs=tol)
                assert eq_t.d2Vdphi2(t, y1_t) == approx(eq_N.d2Vdphi2(N, y1_N), rel=tol, abs=tol)
                assert eq_t.w(t, y1_t) == approx(eq_N.w(N, y1_N), rel=tol, abs=tol)
                assert eq_t.inflating(t, y1_t) == approx(eq_N.inflating(N, y1_N), rel=tol, abs=tol)


@pytest.mark.parametrize('K', [-1, 0, +1])
def test_sol_time_efolds(K):
    pot = QuadraticPotential(Lambda=np.sqrt(6e-6))
    N_i = 10
    phi_i = 17
    t_i = 7e4
    eta_i = 0
    h = 0.7
    k = np.logspace(-3, 1, 4 * 10 + 1)
    Omega_K0 = -K * 0.01

    eq_t = InflationEquationsT(K=K, potential=pot, track_eta=True)
    eq_N = InflationEquationsN(K=K, potential=pot, track_eta=True, track_time=True)
    ic_t = InflationStartIC(eq_t, N_i=N_i, phi_i=phi_i, t_i=t_i, eta_i=eta_i)
    ic_N = InflationStartIC(eq_N, N_i=N_i, phi_i=phi_i, t_i=t_i, eta_i=eta_i)
    ev_t = [InflationEvent(eq_t, +1, terminal=False),
            InflationEvent(eq_t, -1, terminal=True)]
    ev_N = [InflationEvent(eq_N, +1, terminal=False),
            InflationEvent(eq_N, -1, terminal=True)]
    bist = solve(ic=ic_t, events=ev_t, dense_output=True)
    bisn = solve(ic=ic_N, events=ev_N, dense_output=True)
    assert bist.N_tot == approx(bisn.N_tot)

    N2t = interp1d(bisn.N, bisn.t, kind='cubic')
    N2phi = interp1d(bisn.N, bisn.phi, kind='cubic')
    N2H = interp1d(bisn.N, bisn.H, kind='cubic')
    assert_allclose(bist.t[1:-1], N2t(bist.N[1:-1]), rtol=1e-5)
    assert_allclose(bist.phi[1:-1], N2phi(bist.N[1:-1]), rtol=1e-5)
    assert_allclose(bist.H[1:-1], N2H(bist.N[1:-1]), rtol=1e-5)

    bist.derive_a0(Omega_K0=Omega_K0, h=h)
    bisn.derive_a0(Omega_K0=Omega_K0, h=h)
    assert bist.K == K
    assert bisn.K == K
    assert bist.Omega_K0 == Omega_K0
    assert bisn.Omega_K0 == Omega_K0
    if K != 0:
        assert bisn.a0_Mpc * Mpc_m == bisn.a0_lp * lp_m
        bist.derive_comoving_hubble_horizon(Omega_K0=Omega_K0, h=h)
        bisn.derive_comoving_hubble_horizon(Omega_K0=Omega_K0, h=h)
        bist.derive_approx_power(Omega_K0=Omega_K0, h=h)
        bisn.derive_approx_power(Omega_K0=Omega_K0, h=h)
    elif K == 0:
        assert bisn.a0 == 1
        bist.derive_comoving_hubble_horizon(N_star=55)
        bisn.derive_comoving_hubble_horizon(N_star=55)
        bist.derive_approx_power(N_star=55)
        bisn.derive_approx_power(N_star=55)
    assert bist.N_star == approx(bisn.N_star, rel=1e-5)
    assert bist.N_dagg == approx(bisn.N_dagg, rel=1e-5)
    assert bist.n_s == approx(bisn.n_s, rel=1e-5)
    assert bist.n_run == approx(bisn.n_run, rel=1e-3)
    assert bist.r == approx(bisn.r, rel=1e-5)
    assert_allclose(bist.P_s_approx(k) * 1e9, bisn.P_s_approx(k) * 1e9, rtol=1e-4)
    assert_allclose(bist.P_t_approx(k) * 1e9, bisn.P_t_approx(k) * 1e9, rtol=1e-3)


def nan_inflation_end(background_sol):
    assert not np.isfinite(background_sol.N_end)
    assert not np.isfinite(background_sol.phi_end)
    assert not np.isfinite(background_sol.V_end)
    assert not np.isfinite(background_sol.N_tot)
    assert not hasattr(background_sol, 'inflation_mask')


@pytest.mark.parametrize('K', [-1, 0, +1])
@pytest.mark.parametrize('Eq', [InflationEquationsT, InflationEquationsN])
def test_postprocessing_inflation_end_warnings(K, Eq):
    t_i = 1e4
    N_i = 10
    phi_i = 17
    pot = QuadraticPotential(Lambda=np.sqrt(6e-6))
    eq = Eq(K=K, potential=pot, verbose=True)

    # stop at N=20 to trigger "Inflation has not ended." warning:
    ic_early_end = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
    ev = [InflationEvent(ic_early_end.equations, +1, terminal=False),
          InflationEvent(ic_early_end.equations, -1, terminal=True),
          UntilNEvent(ic_early_end.equations, 20)]
    with pytest.warns(InflationEndWarning, match="Still inflating"):
        bist = solve(ic=ic_early_end, events=ev)
    nan_inflation_end(background_sol=bist)

    # no passing of InflationEvent(-1), i.e. inflation end not recorded
    ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
    ev_no_end = [InflationEvent(ic.equations, +1, terminal=False),
                 UntilNEvent(ic.equations, N_i + 65)]
    with pytest.warns(InflationEndWarning, match="Not tracking"):
        bist = solve(ic=ic, events=ev_no_end)
    nan_inflation_end(background_sol=bist)


def test_Ncross_nan():
    pot = QuadraticPotential(Lambda=np.sqrt(6e-6))
    N_i = 18
    phi_i = 15
    t_i = 7e4
    h = 0.7
    for K in [-1, +1]:
        for eq in [InflationEquationsT(K=K, potential=pot),
                   InflationEquationsN(K=K, potential=pot)]:
            Omega_K0 = -K * 0.1
            ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
            ev = [InflationEvent(eq, +1, terminal=False),
                  InflationEvent(eq, -1, terminal=True)]
            b_sol = solve(ic=ic, events=ev)
            print(b_sol.N_tot)
            b_sol.derive_approx_power(Omega_K0=Omega_K0, h=h)
            assert np.log(K_STAR) < np.min(b_sol.logk)
            assert np.isnan(b_sol.N_cross)
            assert np.isnan(b_sol.N_star)
            assert np.isnan(b_sol.N_dagg)


def test_approx_As_ns_nrun_r__with_tolerances_and_slow_roll():
    K = +1
    pot = QuadraticPotential(Lambda=np.sqrt(6e-6))
    t_i = 1e4
    N_i = 10
    Omega_K0 = -K * 0.01
    h = 0.7

    Nstar_range = np.linspace(30, 90, 7)
    rtols = np.array([1e-12, 2.4e-14])
    As_range = np.zeros((rtols.size, Nstar_range.size))
    ns_range = np.zeros((rtols.size, Nstar_range.size))
    nrun_range = np.zeros((rtols.size, Nstar_range.size))
    r_range = np.zeros((rtols.size, Nstar_range.size))

    ns_slow_roll = 1 - 2 / Nstar_range
    r_slow_roll = 8 / Nstar_range

    for i, rtol in enumerate(rtols):
        for j, Nstar in enumerate(Nstar_range):
            eq = InflationEquationsT(K=K, potential=pot)
            ic = ISIC_NsOk(equations=eq, N_i=N_i, N_star=Nstar, Omega_K0=Omega_K0, h=h, t_i=t_i,
                           phi_i_bracket=[15.21, 30])
            ev = [InflationEvent(ic.equations, +1, terminal=False),
                  InflationEvent(ic.equations, -1, terminal=True)]
            bist = solve(ic=ic, events=ev, rtol=rtol)
            bist.derive_approx_power(Omega_K0=Omega_K0, h=h)
            n_s = bist.n_s
            r = bist.r
            assert np.isclose(bist.N_star, Nstar)
            assert np.isclose(n_s, ns_slow_roll[j], rtol=0.005)
            assert np.isclose(r, r_slow_roll[j], rtol=0.005)
            As_range[i, j] = bist.A_s
            ns_range[i, j] = bist.n_s
            nrun_range[i, j] = bist.n_run
            r_range[i, j] = bist.r

    assert_allclose(ns_range[0], ns_slow_roll, rtol=0.005)
    assert_allclose(ns_range[1], ns_slow_roll, rtol=0.005)
    assert_allclose(r_range[0], r_slow_roll, rtol=0.005)
    assert_allclose(r_range[1], r_slow_roll, rtol=0.005)

    assert_allclose(As_range[0], As_range[1], rtol=1e-4, atol=1e-9*1e-3)
    assert_allclose(ns_range[0], ns_range[1], rtol=1e-4)
    assert_allclose(nrun_range[0], nrun_range[1], rtol=1e-4, atol=1e-4)
    assert_allclose(r_range[0], r_range[1], rtol=1e-4)
