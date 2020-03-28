#!/usr/bin/env python
"""Tests for `primpy.inflation` module."""
import pytest
from pytest import approx
import numpy as np
from primpy.potentials import QuadraticPotential
from primpy.events import InflationEvent, UntilNEvent
from primpy.time.initialconditions import InflationStartIC_NiPi, ISIC_msNO
from primpy.inflation import InflationEquations
from primpy.time.inflation import InflationEquationsT
from primpy.efolds.inflation import InflationEquationsN
from primpy.solver import solve


@pytest.mark.xfail(raises=NotImplementedError)
def test_H_not_implemented_error():
    eq = InflationEquations(K=1, potential=QuadraticPotential(mass=6e-6))
    eq.H(x=0, y=np.zeros(4))


@pytest.mark.xfail(raises=NotImplementedError)
def test_H2_not_implemented_error():
    eq = InflationEquations(K=1, potential=QuadraticPotential(mass=6e-6))
    eq.H2(x=0, y=np.zeros(4))


@pytest.mark.xfail(raises=AttributeError)
def test_V_attribute_error():
    eq = InflationEquations(K=1, potential=QuadraticPotential(mass=6e-6))
    eq.V(x=0, y=np.zeros(4))


@pytest.mark.xfail(raises=AttributeError)
def test_dVdphi_attribute_error():
    eq = InflationEquations(K=1, potential=QuadraticPotential(mass=6e-6))
    eq.dVdphi(x=0, y=np.zeros(4))


@pytest.mark.xfail(raises=AttributeError)
def test_d2Vdphi2_attribute_error():
    eq = InflationEquations(K=1, potential=QuadraticPotential(mass=6e-6))
    eq.d2Vdphi2(x=0, y=np.zeros(4))


@pytest.mark.xfail(raises=NotImplementedError)
def test_w_not_implemented_error():
    eq = InflationEquations(K=1, potential=QuadraticPotential(mass=6e-6))
    eq.w(x=0, y=np.zeros(4))


@pytest.mark.xfail(raises=NotImplementedError)
def test_inflating_not_implemented_error():
    eq = InflationEquations(K=1, potential=QuadraticPotential(mass=6e-6))
    eq.inflating(x=0, y=np.zeros(4))


def test_basic_methods_time_vs_efolds():
    tol = 1e-12
    t = 1
    N = 10
    phi = 20
    for K in [-1, 0, 1]:
        for mass in [1, 6e-6]:
            pot = QuadraticPotential(mass=mass)
            for dphidt_squared in [100 * pot.V(phi), 2 * pot.V(phi), pot.V(phi), pot.V(phi) / 100]:
                dphidt = -np.sqrt(dphidt_squared)
                eq_t = InflationEquationsT(K=K, potential=QuadraticPotential(mass=mass))
                eq_N = InflationEquationsN(K=K, potential=QuadraticPotential(mass=mass))
                assert eq_t.idx['N'] == 0
                assert eq_t.idx['phi'] == 1
                assert eq_t.idx['dphidt'] == 2
                assert eq_N.idx['phi'] == 0
                assert eq_N.idx['dphidN'] == 1
                y1_t = np.array([N, phi, dphidt])
                y1_N = np.array([phi, dphidt / eq_t.H(t, y1_t)])
                assert eq_t.H2(t, y1_t) == approx(eq_N.H2(N, y1_N), rel=tol, abs=tol)
                assert eq_t.H(t, y1_t) == approx(eq_N.H(N, y1_N), rel=tol, abs=tol)
                assert eq_t.V(t, y1_t) == approx(eq_N.V(N, y1_N), rel=tol, abs=tol)
                assert eq_t.dVdphi(t, y1_t) == approx(eq_N.dVdphi(N, y1_N), rel=tol, abs=tol)
                assert eq_t.d2Vdphi2(t, y1_t) == approx(eq_N.d2Vdphi2(N, y1_N), rel=tol, abs=tol)
                assert eq_t.w(t, y1_t) == approx(eq_N.w(N, y1_N), rel=tol, abs=tol)
                assert eq_t.inflating(t, y1_t) == approx(eq_N.inflating(N, y1_N), rel=tol, abs=tol)


def nan_inflation_start(background_sol):
    assert not np.isfinite(background_sol.N_beg)
    assert not np.isfinite(background_sol.N_tot)
    assert not hasattr(background_sol, 'inflation_mask')


def test_postprocessing_inflation_start_warnings():
    t_i = 1e4
    N_i = 10
    phi_i = 17
    pot = QuadraticPotential(mass=6e-6)
    for K in [-1, 0, +1]:
        # TODO: add KD initial conditions and test for collapse

        # no passing of InflationEvent(+1), i.e. inflation start not recorded
        ic = InflationStartIC_NiPi(t_i=t_i, N_i=N_i, phi_i=phi_i, K=K, potential=pot)
        ev_no_start = [InflationEvent(ic.equations, -1, terminal=True)]
        with pytest.warns(UserWarning, match="Inflation start not determined. In order to"):
            bist = solve(ic=ic, events=ev_no_start)
        nan_inflation_start(background_sol=bist)

        # no inflation start recorded, despite using event tracking
        # e.g. when integrating backwards in time
        ic_backward = InflationStartIC_NiPi(t_i=t_i, N_i=N_i, phi_i=phi_i, K=K, potential=pot,
                                            t_end=1)
        ev_backward = [UntilNEvent(ic_backward.equations, value=0),
                       InflationEvent(ic_backward.equations, +1, terminal=False)]
        with pytest.warns(UserWarning, match="Inflation start not determined."):
            bist_backward = solve(ic=ic_backward, events=ev_backward)
        nan_inflation_start(background_sol=bist_backward)


def nan_inflation_end(background_sol):
    assert not np.isfinite(background_sol.N_end)
    assert not np.isfinite(background_sol.phi_end)
    assert not np.isfinite(background_sol.V_end)
    assert not np.isfinite(background_sol.N_tot)
    assert not hasattr(background_sol, 'inflation_mask')


def test_postprocessing_inflation_end_warnings():
    t_i = 1e4
    N_i = 10
    phi_i = 17
    pot = QuadraticPotential(mass=6e-6)
    for K in [-1, 0, +1]:
        # set t_end earlier to trigger "inflation not ended warning:
        ic_early_end = InflationStartIC_NiPi(t_i=t_i, N_i=N_i, phi_i=phi_i, K=K, potential=pot,
                                             t_end=1e6)
        ev = [InflationEvent(ic_early_end.equations, +1, terminal=False),
              InflationEvent(ic_early_end.equations, -1, terminal=True)]
        with pytest.warns(UserWarning, match="Inflation has not ended."):
            bist = solve(ic=ic_early_end, events=ev)
        nan_inflation_end(background_sol=bist)

        # no passing of InflationEvent(-1), i.e. inflation end not recorded
        ic = InflationStartIC_NiPi(t_i=t_i, N_i=N_i, phi_i=phi_i, K=K, potential=pot, t_end=1e7)
        ev_no_end = [InflationEvent(ic.equations, +1, terminal=False)]
        with pytest.warns(UserWarning, match="Inflation end not determined. In order to"):
            bist = solve(ic=ic, events=ev_no_end)
        nan_inflation_end(background_sol=bist)

        # no inflation end recorded, despite not being in inflation, despite using event tracking
        # e.g. when integrating backwards in time
        ic_backward = InflationStartIC_NiPi(t_i=t_i, N_i=N_i, phi_i=phi_i, K=K, potential=pot,
                                            t_end=1)
        ev_backward = [UntilNEvent(ic_backward.equations, value=0),
                       InflationEvent(ic_backward.equations, -1, terminal=True)]
        with pytest.warns(UserWarning, match="Inflation end not determined."):
            bist_backward = solve(ic=ic_backward, events=ev_backward)
        nan_inflation_end(background_sol=bist_backward)


def test_approx_As_ns_nrun_r__with_tolerances_and_slow_roll():
    t_i = 1e4
    N_i = 10
    K = +1
    mass = 6e-6
    Omega_K0 = -K * 0.01
    h = 0.7

    Nstar_range = np.linspace(30, 90, 7)
    rtols = np.array([1e-6, 1e-10])
    As_range = np.zeros((rtols.size, Nstar_range.size))
    ns_range = np.zeros((rtols.size, Nstar_range.size))
    nrun_range = np.zeros((rtols.size, Nstar_range.size))
    r_range = np.zeros((rtols.size, Nstar_range.size))

    ns_slow_roll = 1 - 2 / Nstar_range
    r_slow_roll = 8 / Nstar_range

    for i, rtol in enumerate(rtols):
        for j, Nstar in enumerate(Nstar_range):
            print(i, Nstar)
            ic = ISIC_msNO(t_i=t_i, mass=mass, N_star=Nstar, N_i=N_i, Omega_K0=Omega_K0, h=h, K=K,
                           Potential=QuadraticPotential, phi_i_bracket=[15.21, 30])
            ev = [InflationEvent(ic.equations, +1, terminal=False),
                  InflationEvent(ic.equations, -1, terminal=True)]
            bist = solve(ic=ic, events=ev, rtol=1e-10, atol=1e-10)
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

    np.testing.assert_allclose(ns_range[0], ns_slow_roll, rtol=0.005)
    np.testing.assert_allclose(ns_range[1], ns_slow_roll, rtol=0.005)
    np.testing.assert_allclose(r_range[0], r_slow_roll, rtol=0.005)
    np.testing.assert_allclose(r_range[1], r_slow_roll, rtol=0.005)

    np.testing.assert_allclose(As_range[0], As_range[1], rtol=1e-4)
    np.testing.assert_allclose(ns_range[0], ns_range[1], rtol=1e-4)
    np.testing.assert_allclose(nrun_range[0], nrun_range[1], rtol=1e-4)
    np.testing.assert_allclose(r_range[0], r_range[1], rtol=1e-4)
