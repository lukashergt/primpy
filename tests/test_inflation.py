#!/usr/bin/env python
"""Tests for `primpy.inflation` module."""
import pytest
from pytest import approx
from scipy.interpolate import interp1d
import numpy as np
from numpy.testing import assert_allclose
from primpy.exceptionhandling import InflationEndWarning, InsufficientInflationError
from primpy.units import Mpc_m, lp_m
from primpy.parameters import K_STAR
from primpy.potentials import QuadraticPotential, StarobinskyPotential
from primpy.events import InflationEvent, UntilNEvent
from primpy.inflation import InflationEquations
from primpy.time.inflation import InflationEquationsT
from primpy.efolds.inflation import InflationEquationsN
from primpy.initialconditions import InflationStartIC, ISIC_NsOk, SlowRollIC
from primpy.solver import solve


def test_not_implemented_errors():
    eq = InflationEquations(K=1, potential=QuadraticPotential(Lambda=0.0025))
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
            assert hasattr(eq, '_N')
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


@pytest.mark.parametrize('K', [-1, 0, +1])
def test_basic_methods_time_vs_efolds(K):
    tol = 1e-12
    t = 1
    N = 10
    phi = 20
    for Lambda in [1, 0.0025]:
        pot = QuadraticPotential(Lambda=Lambda)
        for dphidt_squared in [100 * pot.V(phi), 2 * pot.V(phi), pot.V(phi), pot.V(phi) / 100]:
            dphidt = -np.sqrt(dphidt_squared)
            eq_t = InflationEquationsT(K=K, potential=pot)
            eq_N = InflationEquationsN(K=K, potential=pot)
            assert eq_t.idx['phi'] == 0
            assert eq_t.idx['dphidt'] == 1
            assert eq_t.idx['_N'] == 2
            assert eq_N.idx['phi'] == 0
            assert eq_N.idx['dphidN'] == 1
            y1_t = np.array([phi, dphidt, N])
            y1_N = np.array([phi, dphidt / eq_t.H(t, y1_t)])
            assert eq_t.H2(t, y1_t) == approx(eq_N.H2(N, y1_N), rel=tol, abs=tol)
            assert eq_t.H(t, y1_t) == approx(eq_N.H(N, y1_N), rel=tol, abs=tol)
            assert eq_t.V(t, y1_t) == approx(eq_N.V(N, y1_N), rel=tol, abs=tol)
            assert eq_t.dVdphi(t, y1_t) == approx(eq_N.dVdphi(N, y1_N), rel=tol, abs=tol)
            assert eq_t.d2Vdphi2(t, y1_t) == approx(eq_N.d2Vdphi2(N, y1_N), rel=tol, abs=tol)
            assert eq_t.w(t, y1_t) == approx(eq_N.w(N, y1_N), rel=tol, abs=tol)
            assert eq_t.inflating(t, y1_t) == approx(eq_N.inflating(N, y1_N), rel=tol, abs=tol)


def test_helper_methods_time_efolds():
    K = 0
    N_star = 60
    N_i = 10
    phi_i = 6
    t_i = 7e4
    k = np.logspace(-3, 1, 4 * 10 + 1)
    Lambda = 0.003
    pot = StarobinskyPotential(Lambda=Lambda)

    eq_t = InflationEquationsT(K=K, potential=pot, track_eta=False)
    eq_N = InflationEquationsN(K=K, potential=pot, track_eta=False, track_time=True)
    ic_t = SlowRollIC(eq_t, phi_i=phi_i, N_i=N_i, t_i=t_i)
    ic_N = SlowRollIC(eq_N, phi_i=phi_i, N_i=N_i, t_i=t_i)
    ev_t = [InflationEvent(eq_t, +1, terminal=False),  # records inflation start
            InflationEvent(eq_t, -1, terminal=True)]   # records inflation end
    ev_N = [InflationEvent(eq_N, +1, terminal=False),  # records inflation start
            InflationEvent(eq_N, -1, terminal=True)]   # records inflation end
    bsrt = solve(ic=ic_t, events=ev_t, dense_output=True, method='DOP853', rtol=1e-12)
    bsrN = solve(ic=ic_N, events=ev_N, dense_output=True, method='DOP853', rtol=1e-12)
    bsrt.calibrate_scale_factor(N_star=N_star)
    bsrN.calibrate_scale_factor(N_star=N_star)

    N = np.arange(bsrt._N_end-N_star, bsrt._N_end-N_star/2, 1)
    bt_H = bsrt.H
    bN_H = bsrN.H
    bt_dphi = bsrt.dphidt
    bN_dphi = bsrN.dphidN
    bt_dH_H = eq_t.get_dH_H(bsrt._N, bt_H**2, bt_dphi, K=K)
    bN_dH_H = eq_N.get_dH_H(bsrN._N, bN_H**2, bN_dphi, K=K)
    bt_dH = eq_t.get_dH(bsrt._N, bt_H, bt_dphi, K=K)
    bN_dH = eq_N.get_dH(bsrN._N, bN_H, bN_dphi, K=K)
    bt_dV = bsrt.potential.dV(bsrt.phi)
    bN_dV = bsrN.potential.dV(bsrN.phi)
    bt_d2V = bsrt.potential.d2V(bsrt.phi)
    bN_d2V = bsrN.potential.d2V(bsrN.phi)
    bt_d3V = bsrt.potential.d3V(bsrt.phi)
    bN_d3V = bsrN.potential.d3V(bsrN.phi)
    bt_d2phi = eq_t.get_d2phi(bt_H**2, bt_dH/bt_H, bt_dphi, bt_dV)
    bN_d2phi = eq_N.get_d2phi(bN_H**2, bN_dH/bN_H, bN_dphi, bN_dV)
    bt_d2H = eq_t.get_d2H(bsrt._N, bt_H, bt_dH, bt_dphi, bt_d2phi, K=K)
    bN_d2H = eq_N.get_d2H(bsrN._N, bN_H, bN_dH, bN_dphi, bN_d2phi, K=K)
    bt_d3phi = eq_t.get_d3phi(bt_H, bt_dH, bt_d2H, bt_dphi, bt_d2phi, bt_dV, bt_d2V)
    bN_d3phi = eq_N.get_d3phi(bN_H, bN_dH, bN_d2H, bN_dphi, bN_d2phi, bN_dV, bN_d2V)
    bt_d3H = eq_t.get_d3H(bsrt._N, bt_H, bt_dH, bt_d2H, bt_dphi, bt_d2phi, bt_d3phi, K=K)
    bN_d3H = eq_N.get_d3H(bsrN._N, bN_H, bN_dH, bN_d2H, bN_dphi, bN_d2phi, bN_d3phi, K=K)
    bt_d4phi = eq_t.get_d4phi(bt_H, bt_dH, bt_d2H, bt_d3H, bt_dphi, bt_d2phi, bt_d3phi, bt_dV, bt_d2V, bt_d3V)  # noqa: E501
    bN_d4phi = eq_N.get_d4phi(bN_H, bN_dH, bN_d2H, bN_d3H, bN_dphi, bN_d2phi, bN_d3phi, bN_dV, bN_d2V, bN_d3V)  # noqa: E501
    bt_e1 = eq_t.get_epsilon_1H(bt_H, bt_dH)
    bN_e1 = eq_N.get_epsilon_1H(bN_H, bN_dH)
    bt_e2 = eq_t.get_epsilon_2H(bt_H, bt_dH, bt_d2H)
    bN_e2 = eq_N.get_epsilon_2H(bN_H, bN_dH, bN_d2H)
    bt_e3 = eq_t.get_epsilon_3H(bt_H, bt_dH, bt_d2H, bt_d3H)
    bN_e3 = eq_N.get_epsilon_3H(bN_H, bN_dH, bN_d2H, bN_d3H)
    bt_e2_Gong = eq_t.get_epsilon_2H(bt_H, bt_dH, bt_d2H, kind='Gong')
    bN_e2_Gong = eq_N.get_epsilon_2H(bN_H, bN_dH, bN_d2H, kind='Gong')
    bt_e3_Gong = eq_t.get_epsilon_3H(bt_H, bt_dH, bt_d2H, bt_d3H, kind='Gong')
    bN_e3_Gong = eq_N.get_epsilon_3H(bN_H, bN_dH, bN_d2H, bN_d3H, kind='Gong')

    N_to_H_t = interp1d(bsrt._N, bt_H, kind='cubic')
    N_to_H_N = interp1d(bsrN._N, bN_H, kind='cubic')
    N_to_dH_H_t = interp1d(bsrt._N, bt_dH_H, kind='cubic')
    N_to_dH_H_N = interp1d(bsrN._N, bN_dH_H, kind='cubic')
    N_to_dH_t = interp1d(bsrt._N, bt_dH, kind='cubic')
    N_to_dH_N = interp1d(bsrN._N, bN_dH, kind='cubic')
    N_to_d2H_t = interp1d(bsrt._N, bt_d2H, kind='cubic')
    N_to_d2H_N = interp1d(bsrN._N, bN_d2H, kind='cubic')
    N_to_d3H_t = interp1d(bsrt._N, bt_d3H, kind='cubic')
    N_to_d3H_N = interp1d(bsrN._N, bN_d3H, kind='cubic')
    N_to_dphi_t = interp1d(bsrt._N, bt_dphi, kind='cubic')
    N_to_dphi_N = interp1d(bsrN._N, bN_dphi, kind='cubic')
    N_to_d2phi_t = interp1d(bsrt._N, bt_d2phi, kind='cubic')
    N_to_d2phi_N = interp1d(bsrN._N, bN_d2phi, kind='cubic')
    N_to_d3phi_t = interp1d(bsrt._N, bt_d3phi, kind='cubic')
    N_to_d3phi_N = interp1d(bsrN._N, bN_d3phi, kind='cubic')
    N_to_d4phi_t = interp1d(bsrt._N, bt_d4phi, kind='cubic')
    N_to_d4phi_N = interp1d(bsrN._N, bN_d4phi, kind='cubic')
    N_to_e1_t = interp1d(bsrt._N, bt_e1, kind='cubic')
    N_to_e1_N = interp1d(bsrN._N, bN_e1, kind='cubic')
    N_to_e2_t = interp1d(bsrt._N, bt_e2, kind='cubic')
    N_to_e2_N = interp1d(bsrN._N, bN_e2, kind='cubic')
    N_to_e3_t = interp1d(bsrt._N, bt_e3, kind='cubic')
    N_to_e3_N = interp1d(bsrN._N, bN_e3, kind='cubic')
    N_to_e2_Gong_t = interp1d(bsrt._N, bt_e2_Gong, kind='cubic')
    N_to_e2_Gong_N = interp1d(bsrN._N, bN_e2_Gong, kind='cubic')
    N_to_e3_Gong_t = interp1d(bsrt._N, bt_e3_Gong, kind='cubic')
    N_to_e3_Gong_N = interp1d(bsrN._N, bN_e3_Gong, kind='cubic')

    H_t = N_to_H_t(N)
    H_N = N_to_H_N(N)
    dHdt_H = N_to_dH_H_t(N)
    dHdN_H = N_to_dH_H_N(N)
    dHdt = N_to_dH_t(N)
    dHdN = N_to_dH_N(N)
    d2Hdt2 = N_to_d2H_t(N)
    d2HdN2 = N_to_d2H_N(N)
    d3Hdt3 = N_to_d3H_t(N)
    d3HdN3 = N_to_d3H_N(N)
    dphidt = N_to_dphi_t(N)
    dphidN = N_to_dphi_N(N)
    d2phidt2 = N_to_d2phi_t(N)
    d2phidN2 = N_to_d2phi_N(N)
    d3phidt3 = N_to_d3phi_t(N)
    d3phidN3 = N_to_d3phi_N(N)
    d4phidt4 = N_to_d4phi_t(N)
    d4phidN4 = N_to_d4phi_N(N)
    e1_t = N_to_e1_t(N)
    e1_N = N_to_e1_N(N)
    e2_t = N_to_e2_t(N)
    e2_N = N_to_e2_N(N)
    e3_t = N_to_e3_t(N)
    e3_N = N_to_e3_N(N)
    e2_Gong_t = N_to_e2_Gong_t(N)
    e2_Gong_N = N_to_e2_Gong_N(N)
    e3_Gong_t = N_to_e3_Gong_t(N)
    e3_Gong_N = N_to_e3_Gong_N(N)

    assert_allclose(H_t, H_N, rtol=1e-8)
    assert_allclose(dHdt_H, dHdN, rtol=1e-6)
    assert_allclose(dHdt_H, dHdN_H * H_N, rtol=1e-6)
    assert_allclose(dHdt, dHdN * H_N, rtol=1e-6)
    assert_allclose(d2Hdt2, d2HdN2 * H_N**2 + dHdN**2 * H_N, rtol=1e-6)
    assert_allclose(d3Hdt3, d3HdN3*H_N**3 + 4*d2HdN2*dHdN*H_N**2 + dHdN**3*H_N, rtol=1e-6)
    assert_allclose(dphidt, dphidN * H_N, rtol=1e-6)
    assert_allclose(d2phidt2, d2phidN2 * H_N**2 + dphidN * dHdN * H_N, rtol=1e-6)
    assert_allclose(d3phidt3, d3phidN3*H_N**3 + 3*d2phidN2*dHdN*H_N**2 + dphidN*d2HdN2*H_N**2 + dphidN*dHdN**2*H_N, rtol=1e-6)  # noqa: E501
    assert_allclose(d4phidt4, d4phidN4*H_N**4 + 6*d3phidN3*dHdN*H_N**3 + 4*d2phidN2*d2HdN2*H_N**3 + dphidN*d3HdN3*H_N**3 + 7*d2phidN2*dHdN**2*H_N**2 + 4*dphidN*d2HdN2*dHdN*H_N**2 + dphidN*dHdN**3*H_N, rtol=2e-5)  # noqa: E501
    assert_allclose(e1_t, e1_N, rtol=1e-6)
    assert_allclose(e2_t, e2_N, rtol=1e-6)
    assert_allclose(e3_t, e3_N, rtol=1e-6)
    assert_allclose(e2_Gong_t, e2_Gong_N, rtol=1e-6)
    assert_allclose(e3_Gong_t, e3_Gong_N, rtol=1e-6)


@pytest.mark.parametrize('K', [-1, 0, +1])
def test_sol_time_efolds(K):
    pot = QuadraticPotential(Lambda=0.0025)
    N_i = 10
    phi_i = 17
    t_i = 7e4
    eta_i = 0
    h = 0.7
    k = np.logspace(-3, 1, 4 * 10 + 1)
    Omega_K0 = -K * 0.001

    eq_t = InflationEquationsT(K=K, potential=pot, track_eta=True)
    eq_N = InflationEquationsN(K=K, potential=pot, track_eta=True, track_time=True)
    ic_t = InflationStartIC(eq_t, N_i=N_i, phi_i=phi_i, t_i=t_i, eta_i=eta_i)
    ic_N = InflationStartIC(eq_N, N_i=N_i, phi_i=phi_i, t_i=t_i, eta_i=eta_i)
    ev_t = [InflationEvent(eq_t, +1, terminal=False),
            InflationEvent(eq_t, -1, terminal=True)]
    ev_N = [InflationEvent(eq_N, +1, terminal=False),
            InflationEvent(eq_N, -1, terminal=True)]
    bist = solve(ic=ic_t, events=ev_t, dense_output=True, method='DOP853', rtol=1e-13, atol=1e-18)
    bisn = solve(ic=ic_N, events=ev_N, dense_output=True, method='DOP853', rtol=1e-13, atol=1e-18)
    assert bist.N_tot == approx(bisn.N_tot, rel=1e-5)

    N2t = interp1d(bisn._N, bisn.t, kind='cubic')
    N2phi = interp1d(bisn._N, bisn.phi, kind='cubic')
    N2H = interp1d(bisn._N, bisn.H, kind='cubic')
    assert_allclose(bist.t[1:-1], N2t(bist._N[1:-1]), rtol=1e-5)
    assert_allclose(bist.phi[1:-1], N2phi(bist._N[1:-1]), rtol=1e-4)
    assert_allclose(bist.H[1:-1], N2H(bist._N[1:-1]), rtol=1e-4)

    # using Omega_K0 or N_star
    bist.calibrate_scale_factor(Omega_K0=Omega_K0, h=h, N_star=55 if K == 0 else None)
    bisn.calibrate_scale_factor(Omega_K0=Omega_K0, h=h, N_star=55 if K == 0 else None)
    assert bist.K == K
    assert bisn.K == K
    assert bist.Omega_K0 == Omega_K0
    assert bisn.Omega_K0 == Omega_K0
    if K != 0:
        assert bisn.a0_Mpc * Mpc_m == approx(bisn.a0 * lp_m, rel=1e-14)
    elif K == 0:
        assert bisn.a0 == 1
    assert bist.N_star == approx(bisn.N_star, rel=1e-5)
    assert bist.N_dagg == approx(bisn.N_dagg, rel=1e-5)
    assert bist.A_s == approx(bisn.A_s, rel=1e-8)
    assert bist.n_s == approx(bisn.n_s, rel=1e-5)
    assert bist.n_run == approx(bisn.n_run, rel=1e-3)
    assert bist.n_runrun == approx(bisn.n_runrun, rel=1e-2)#, abs=1e-6)
    assert bist.A_t == approx(bisn.A_t, rel=1e-8)
    assert bist.r == approx(bisn.r, rel=1e-5)
    assert bist.n_t == approx(bisn.n_t, rel=1e-5)
    assert_allclose(bist.logk2logP_s(np.log(k)), bisn.logk2logP_s(np.log(k)), rtol=1e-6)
    assert_allclose(bist.logk2logP_t(np.log(k)), bisn.logk2logP_t(np.log(k)), rtol=1e-6)
    assert_allclose(bist.P_s_approx(k) * 1e9, bisn.P_s_approx(k) * 1e9, rtol=1e-4)
    assert_allclose(bist.P_t_approx(k) * 1e9, bisn.P_t_approx(k) * 1e9, rtol=1e-3)

    # reheating
    bist.calibrate_scale_factor(calibration_method='reheating', h=h, delta_reh=2, w_reh=0)
    bisn.calibrate_scale_factor(calibration_method='reheating', h=h, delta_reh=2, w_reh=0)
    assert bist.N_star == approx(bisn.N_star, rel=1e-5)
    assert bist.N_dagg == approx(bisn.N_dagg, rel=1e-5)
    assert bist.A_s == approx(bisn.A_s, rel=1e-8)
    assert bist.n_s == approx(bisn.n_s, rel=1e-5)
    assert bist.n_run == approx(bisn.n_run, rel=1e-3)
    assert bist.n_runrun == approx(bisn.n_runrun, rel=1e-2)
    assert bist.A_t == approx(bisn.A_t, rel=1e-8)
    assert bist.r == approx(bisn.r, rel=1e-5)
    assert bist.n_t == approx(bisn.n_t, rel=1e-5)
    assert_allclose(bist.logk2logP_s(np.log(k)), bisn.logk2logP_s(np.log(k)), rtol=1e-6)
    assert_allclose(bist.logk2logP_t(np.log(k)), bisn.logk2logP_t(np.log(k)), rtol=1e-6)
    assert_allclose(bist.P_s_approx(k) * 1e9, bisn.P_s_approx(k) * 1e9, rtol=1e-4)
    assert_allclose(bist.P_t_approx(k) * 1e9, bisn.P_t_approx(k) * 1e9, rtol=1e-3)

    # logaH_star
    if K == 0:
        bist.calibrate_scale_factor(background=bist, N_star=bist.N_star)
        bisn.calibrate_scale_factor(background=bist, N_star=bist.N_star)
        assert bist.N_star == approx(bisn.N_star, rel=1e-5)
        assert bist.N_dagg == approx(bisn.N_dagg, rel=1e-5)
        assert bist.A_s == approx(bisn.A_s, rel=1e-8)
        assert bist.n_s == approx(bisn.n_s, rel=1e-5)
        assert bist.n_run == approx(bisn.n_run, rel=1e-3)
        assert bist.n_runrun == approx(bisn.n_runrun, rel=1e-2)
        assert bist.A_t == approx(bisn.A_t, rel=1e-8)
        assert bist.r == approx(bisn.r, rel=1e-5)
        assert bist.n_t == approx(bisn.n_t, rel=1e-5)
        assert_allclose(bist.logk2logP_s(np.log(k)), bisn.logk2logP_s(np.log(k)), rtol=1e-6)
        assert_allclose(bist.logk2logP_t(np.log(k)), bisn.logk2logP_t(np.log(k)), rtol=1e-6)
        assert_allclose(bist.P_s_approx(k) * 1e9, bisn.P_s_approx(k) * 1e9, rtol=1e-4)
        assert_allclose(bist.P_t_approx(k) * 1e9, bisn.P_t_approx(k) * 1e9, rtol=1e-3)


@pytest.mark.parametrize('K', [-1, 0, +1])
@pytest.mark.parametrize('delta_reh', [0, 5])
@pytest.mark.parametrize('w_reh', [-1/3, 0, 1/3, 1])
def test_reheating(K, delta_reh, w_reh):
    pot = StarobinskyPotential(Lambda=3.3e-3)
    N_i = 14
    phi_i = 6
    t_i = 7e4
    h = 0.7
    eq_t = InflationEquationsT(K=K, potential=pot)
    eq_N = InflationEquationsN(K=K, potential=pot, track_time=True)
    ic_t = InflationStartIC(eq_t, N_i=N_i, phi_i=phi_i, t_i=t_i)
    ic_N = InflationStartIC(eq_N, N_i=N_i, phi_i=phi_i, t_i=t_i)
    ev_t = [InflationEvent(eq_t, +1, terminal=False),
            InflationEvent(eq_t, -1, terminal=True)]
    ev_N = [InflationEvent(eq_N, +1, terminal=False),
            InflationEvent(eq_N, -1, terminal=True)]
    bist = solve(ic=ic_t, events=ev_t, dense_output=True, method='DOP853', rtol=1e-12)
    bisn = solve(ic=ic_N, events=ev_N, dense_output=True, method='DOP853', rtol=1e-12)
    bist.calibrate_scale_factor(calibration_method='reheating', h=h,
                                delta_reh=delta_reh, w_reh=w_reh)
    bisn.calibrate_scale_factor(calibration_method='reheating', h=h,
                                delta_reh=delta_reh, w_reh=w_reh)
    assert bist.N_star == approx(bisn.N_star, rel=1e-5)
    assert bist.N_dagg == approx(bisn.N_dagg, rel=1e-5)
    assert bist.N_reh == approx(bisn.N_reh, rel=1e-5)


@pytest.mark.parametrize('K', [-1, 0, +1])
@pytest.mark.parametrize('rho_reh_GeV', [1e3, 1e9])
def test_derived_reheating(K, rho_reh_GeV):
    pot = StarobinskyPotential(Lambda=3.3e-3)
    N_i = 14
    phi_i = 6
    t_i = 7e4
    h = 0.7
    Omega_K0 = -0.1 * K
    N_star = 55
    eq_t = InflationEquationsT(K=K, potential=pot)
    ic_t = InflationStartIC(eq_t, N_i=N_i, phi_i=phi_i, t_i=t_i)
    ev_t = [InflationEvent(eq_t, +1, terminal=False),
            InflationEvent(eq_t, -1, terminal=True)]
    bist = solve(ic=ic_t, events=ev_t, dense_output=True, method='DOP853', rtol=1e-12)
    bist.calibrate_scale_factor(Omega_K0=Omega_K0, h=h, N_star=N_star, rho_reh_GeV=rho_reh_GeV)
    assert bist.rho_reh_GeV == rho_reh_GeV
    assert bist.N_end < bist.N_reh
    assert -1/3 <= bist.w_reh <= 1
    assert bist.delta_reh >= 0


def nan_inflation_end(background_sol):
    assert not np.isfinite(background_sol._N_end)
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
    pot = QuadraticPotential(Lambda=0.0025)
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


@pytest.mark.parametrize('K', [-1, 0, +1])
@pytest.mark.parametrize('Eq', [InflationEquationsT, InflationEquationsN])
def test_Ncross_not_during_inflation(K, Eq):
    pot = QuadraticPotential(Lambda=0.0025)
    N_i = 18
    phi_i = 15
    t_i = 7e4
    h = 0.7
    N_star = 55
    eq = Eq(K=K, potential=pot)
    Omega_K0 = -K * 0.1
    ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
    ev = [InflationEvent(eq, +1, terminal=False),
          InflationEvent(eq, -1, terminal=True)]
    b_sol = solve(ic=ic, events=ev)
    with pytest.raises(InsufficientInflationError):
        b_sol.calibrate_scale_factor(Omega_K0=Omega_K0, h=h, N_star=N_star if K == 0 else None)
    if K == 0:
        assert b_sol.N_tot < N_star
    else:
        assert np.log(K_STAR) < np.min(b_sol.logk)
    with pytest.raises(InsufficientInflationError):
        b_sol.calibrate_scale_factor(calibration_method='reheating', h=h, delta_reh=5, w_reh=0)
    if K == 0:
        assert b_sol._logaH_star < b_sol._logaH_beg
    else:
        assert np.log(K_STAR) < np.min(b_sol.logk)


def test_calibration_input_errors():
    N_i = 10
    phi_i = 20
    t_i = 7e4
    h = 0.7
    pot = QuadraticPotential(Lambda=0.0025)

    # flat universe
    N_star = 55
    eq = InflationEquationsT(K=0, potential=pot)
    ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
    ev = [InflationEvent(eq, +1, terminal=False),
          InflationEvent(eq, -1, terminal=True)]
    b = solve(ic=ic, events=ev)
    b.calibrate_scale_factor(N_star=N_star)
    b_sol = solve(ic=ic, events=ev)
    with pytest.raises(ValueError):
        b_sol.calibrate_scale_factor(N_star=N_star, Omega_K0=-0.1, h=h)
    with pytest.raises(ValueError):
        b_sol.calibrate_scale_factor(N_star=None)
    with pytest.raises(ValueError):
        b_sol.calibrate_scale_factor(N_star=-N_star)
    with pytest.raises(ValueError):
        b_sol.calibrate_scale_factor(calibration_method='reheating', w_reh=0, delta_reh=-5)
    with pytest.raises(ValueError):
        b_sol.calibrate_scale_factor(calibration_method='reheating', w_reh=-1, delta_reh=5)
    with pytest.raises(ValueError):
        b_sol.calibrate_scale_factor(calibration_method='reheating', w_reh=0, delta_reh=None)
    with pytest.raises(ValueError):
        b_sol.calibrate_scale_factor(calibration_method='reheating', w_reh=None, delta_reh=5)
    with pytest.raises(ValueError):
        b_sol.calibrate_scale_factor(background=b, N_star=None)
    with pytest.raises(ValueError):
        b_sol.calibrate_scale_factor(background=b, N_star=-N_star)
    with pytest.raises(ValueError):
        b_sol.calibrate_scale_factor(background=b, N_star=50)
    with pytest.raises(NotImplementedError):
        b_sol.calibrate_scale_factor(calibration_method='spam')

    # curved universe
    K = 1
    Omega_K0 = -K * 0.1
    eq = InflationEquationsT(K=K, potential=pot)
    ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
    ev = [InflationEvent(eq, +1, terminal=False),
          InflationEvent(eq, -1, terminal=True)]
    b_sol = solve(ic=ic, events=ev)
    with pytest.raises(ValueError):  # missing h
        b_sol.calibrate_scale_factor(Omega_K0=Omega_K0)
    with pytest.raises(ValueError):  # negative h
        b_sol.calibrate_scale_factor(Omega_K0=Omega_K0, h=-h)
    with pytest.raises(ValueError):  # missing Omega_K0
        b_sol.calibrate_scale_factor(h=h)
    with pytest.raises(ValueError):  # wrong Omega_K0
        b_sol.calibrate_scale_factor(h=h, Omega_K0=0)
    with pytest.raises(ValueError):  # Omega_K0 and K do not match
        b_sol.calibrate_scale_factor(h=h, Omega_K0=-Omega_K0)
    with pytest.raises(ValueError):  # Omega_K0 should be None for calibration_method='reheating'
        b_sol.calibrate_scale_factor(calibration_method='reheating', h=h, Omega_K0=Omega_K0)
    with pytest.raises(ValueError):  # negative delta_reh
        b_sol.calibrate_scale_factor(calibration_method='reheating', h=h, w_reh=0, delta_reh=-5)
    with pytest.raises(ValueError):  # w_reh < -1/3
        b_sol.calibrate_scale_factor(calibration_method='reheating', h=h, w_reh=-1, delta_reh=5)
    with pytest.raises(ValueError):  # w_reh provided but delta_reh missing
        b_sol.calibrate_scale_factor(calibration_method='reheating', h=h, w_reh=0, delta_reh=None)
    with pytest.raises(ValueError):  # delta_reh provided but w_reh missing
        b_sol.calibrate_scale_factor(calibration_method='reheating', h=h, w_reh=None, delta_reh=5)
    with pytest.raises(NotImplementedError):  # non-existent calibration_method
        b_sol.calibrate_scale_factor(calibration_method='spam', h=h)


@pytest.mark.parametrize('N_star', [30, 90])
def test_approx_As_ns_nrun_r__with_tolerances_and_slow_roll(N_star):
    K = +1
    pot = QuadraticPotential(Lambda=0.0025)
    t_i = 1e4
    N_i = 10
    Omega_K0 = -K * 0.01
    h = 0.7

    rtols = np.array([1e-12, 2.4e-14])
    As_range = np.zeros(rtols.size)
    ns_range = np.zeros(rtols.size)
    nrun_range = np.zeros(rtols.size)
    r_range = np.zeros(rtols.size)

    ns_slow_roll = 1 - 2 / N_star
    r_slow_roll = 8 / N_star

    for i, rtol in enumerate(rtols):
        eq = InflationEquationsT(K=K, potential=pot)
        ic = ISIC_NsOk(equations=eq, N_i=N_i, N_star=N_star, Omega_K0=Omega_K0, h=h, t_i=t_i,
                       phi_i_bracket=[12, 30])
        ev = [InflationEvent(ic.equations, +1, terminal=False),
              InflationEvent(ic.equations, -1, terminal=True)]
        bist = solve(ic=ic, events=ev, rtol=rtol)
        bist.calibrate_scale_factor(Omega_K0=Omega_K0, h=h)
        n_s = bist.n_s
        r = bist.r
        assert np.isclose(bist.N_star, N_star)
        assert np.isclose(n_s, ns_slow_roll, rtol=0.005)
        assert np.isclose(r, r_slow_roll, rtol=0.005)
        As_range[i] = bist.A_s
        ns_range[i] = bist.n_s
        nrun_range[i] = bist.n_run
        r_range[i] = bist.r

    assert_allclose(ns_range[0], ns_slow_roll, rtol=0.005)
    assert_allclose(ns_range[1], ns_slow_roll, rtol=0.005)
    assert_allclose(r_range[0], r_slow_roll, rtol=0.005)
    assert_allclose(r_range[1], r_slow_roll, rtol=0.005)

    assert_allclose(As_range[0], As_range[1], rtol=1e-4, atol=1e-9*1e-3)
    assert_allclose(ns_range[0], ns_range[1], rtol=1e-4)
    assert_allclose(nrun_range[0], nrun_range[1], rtol=1e-4, atol=1e-4)
    assert_allclose(r_range[0], r_range[1], rtol=1e-4)
