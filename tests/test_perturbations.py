#!/usr/bin/env python
"""Tests for `primpy.perturbation` module."""
import pytest
from pytest import approx
import numpy as np
from numpy.testing import assert_allclose
from primpy.potentials import QuadraticPotential
from primpy.events import InflationEvent, CollapseEvent
from primpy.time.inflation import InflationEquationsT
from primpy.efolds.inflation import InflationEquationsN
from primpy.initialconditions import InflationStartIC
from primpy.time.perturbations import CurvaturePerturbationT
from primpy.efolds.perturbations import CurvaturePerturbationN
from primpy.solver import solve
from primpy.solver import solve_oscode


def setup_background(K, f_i, abs_Omega_K0):
    pot = QuadraticPotential(mass=6e-6)
    phi_i = 16
    t_i = 5e4
    Omega_K0 = -K * abs_Omega_K0
    Omega_Ki = f_i * Omega_K0
    h = 0.7

    eq_t = InflationEquationsT(K=K, potential=pot)
    eq_n = InflationEquationsN(K=K, potential=pot)
    ic_t = InflationStartIC(eq_t, phi_i=phi_i, Omega_Ki=Omega_Ki, t_i=t_i)
    ic_n = InflationStartIC(eq_n, phi_i=phi_i, Omega_Ki=Omega_Ki, t_i=None)
    ev_t = [InflationEvent(eq_t, +1, terminal=False),
            InflationEvent(eq_t, -1, terminal=True),
            CollapseEvent(eq_t)]
    ev_n = [InflationEvent(eq_n, +1, terminal=False),
            InflationEvent(eq_n, -1, terminal=True),
            CollapseEvent(eq_n)]
    t_eval = np.linspace(t_i, 5e6, int(1e5))
    N_eval = np.linspace(ic_n.N_i, 100, int(1e5))
    bist = solve(ic=ic_t, events=ev_t, t_eval=t_eval, dense_output=True, rtol=1e-10, atol=1e-10)
    bisn = solve(ic=ic_n, events=ev_n, t_eval=N_eval, dense_output=True, rtol=1e-12, atol=1e-12)
    assert bist.independent_variable == 't'
    assert bisn.independent_variable == 'N'
    assert bist.N_tot == approx(bisn.N_tot)
    bist.derive_a0(Omega_K0=Omega_K0, h=h)
    bisn.derive_a0(Omega_K0=Omega_K0, h=h)
    assert bist.a0_Mpc == approx(bisn.a0_Mpc)
    bist.derive_approx_power(Omega_K0=Omega_K0, h=h)
    bisn.derive_approx_power(Omega_K0=Omega_K0, h=h)
    assert bist.N_star == approx(bisn.N_star)

    return bist, bisn


@pytest.mark.parametrize('K', [-1, +1])
@pytest.mark.parametrize('f_i', [10])  # FIXME: make 100, 1000 work as well
@pytest.mark.parametrize('abs_Omega_K0', [0.09, 0.009])
def test_background_setup(K, f_i, abs_Omega_K0):
    if -K * f_i * abs_Omega_K0 >= 1:
        with pytest.raises(Exception):
            setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
    else:
        setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)


# noinspection DuplicatedCode
@pytest.mark.parametrize('K', [-1, +1])
@pytest.mark.parametrize('f_i', [10])  # FIXME: make 100, 1000 work as well
@pytest.mark.parametrize('abs_Omega_K0', [0.09, 0.009])
@pytest.mark.parametrize('k_iMpc', np.logspace(-6, 0, 6 + 1))
def test_perturbations_frequency_damping(K, f_i, abs_Omega_K0, k_iMpc):
    if -K * f_i * abs_Omega_K0 >= 1:
        with pytest.raises(Exception):
            setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
    else:
        bist, bisn = setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
        k = k_iMpc * bist.a0_Mpc
        pert_t = CurvaturePerturbationT(background=bist, k=k)
        pert_n = CurvaturePerturbationN(background=bisn, k=k)
        with pytest.raises(NotImplementedError):
            pert_t(bist.x[0], bist.y[0])
        with pytest.raises(NotImplementedError):
            pert_n(bisn.x[0], bisn.y[0])
        freq_t, damp_t = pert_t.mukhanov_sasaki_frequency_damping(background=bist, k=k)
        freq_n, damp_n = pert_n.mukhanov_sasaki_frequency_damping(background=bisn, k=k)
        assert np.all(freq_t > 0)
        assert np.all(freq_n > 0)
        assert np.isfinite(damp_t).all()
        assert np.isfinite(damp_n).all()

        pert_t = solve_oscode(bist, k, rtol=1e-5)
        pert_n = solve_oscode(bisn, k, rtol=1e-5)
        assert np.all(np.isfinite(pert_t.one.t))
        assert np.all(np.isfinite(pert_t.two.t))
        assert np.all(np.isfinite(pert_n.one.N))
        assert np.all(np.isfinite(pert_n.two.N))
        assert np.all(np.isfinite(pert_t.one.Rk))
        assert np.all(np.isfinite(pert_t.two.Rk))
        assert np.all(np.isfinite(pert_n.one.Rk))
        assert np.all(np.isfinite(pert_n.two.Rk))
        assert np.all(np.isfinite(pert_t.one.dRk))
        assert np.all(np.isfinite(pert_t.two.dRk))
        assert np.all(np.isfinite(pert_n.one.dRk))
        assert np.all(np.isfinite(pert_n.two.dRk))
        assert pert_n.PPS_RST == approx(pert_t.PPS_RST)


@pytest.mark.parametrize('K', [-1, +1])
@pytest.mark.parametrize('f_i', [10])  # FIXME: make 100, 1000 work as well
@pytest.mark.parametrize('abs_Omega_K0', [0.09, 0.009])
def test_perturbations_discrete_time_efolds(K, f_i, abs_Omega_K0):
    if -K * f_i * abs_Omega_K0 >= 1:
        with pytest.raises(Exception):
            setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
    else:
        bist, bisn = setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
        rtol = 1e-3
        atol = 1e-5
        ks_disc = np.arange(1, 50, 1)  # FIXME: make this work up to 100 at least?
        pps_disc_t = solve_oscode(background=bist, k=ks_disc, rtol=1e-5) * 1e9
        pps_disc_n = solve_oscode(background=bisn, k=ks_disc, rtol=1e-5) * 1e9
        assert np.isfinite(pps_disc_t).all()
        assert np.isfinite(pps_disc_n).all()
        assert_allclose(pps_disc_t, pps_disc_n, rtol=rtol, atol=atol)


@pytest.mark.parametrize('K', [-1, +1])
@pytest.mark.parametrize('f_i', [10])  # FIXME: make 100, 1000 work as well
@pytest.mark.parametrize('Omega_K0', [0.09, 0.009])
def test_perturbations_continuous_time_vs_efolds(K, f_i, Omega_K0):
    if -K * f_i * Omega_K0 >= 1:
        with pytest.raises(Exception):
            setup_background(K=K, f_i=f_i, Omega_K0=Omega_K0)
    else:
        bist, bisn = setup_background(K=K, f_i=f_i, Omega_K0=Omega_K0)
        rtol = 5e-3
        atol = 1e-5
        ks_iMpc = np.logspace(-4, -1, 3 * 10 + 1)  # FIXME: make this work for smaller k?
        ks_cont = ks_iMpc * bist.a0_Mpc
        pps_cont_t = solve_oscode(background=bist, k=ks_cont, rtol=1e-5) * 1e9
        pps_cont_n = solve_oscode(background=bisn, k=ks_cont, rtol=1e-5) * 1e9
        mask = np.isfinite(pps_cont_t) & np.isfinite(pps_cont_n)  # FIXME: manage without masking
        # assert np.isfinite(pps_cont_t).all()
        # assert np.isfinite(pps_cont_n).all()
        assert_allclose(pps_cont_t[mask], pps_cont_n[mask], rtol=rtol, atol=atol)


@pytest.mark.parametrize('K', [-1, +1])
@pytest.mark.parametrize('f_i', [10])  # FIXME: make 100, 1000 work as well
@pytest.mark.parametrize('Omega_K0', [0.09, 0.009])
def test_perturbations_large_scales_pyoscode_vs_background(K, f_i, Omega_K0):
    if -K * f_i * Omega_K0 >= 1:
        with pytest.raises(Exception):
            setup_background(K=K, f_i=f_i, Omega_K0=Omega_K0)
    else:
        bist, bisn = setup_background(K=K, f_i=f_i, Omega_K0=Omega_K0)
        rtol = 0.05
        atol = 1e-5
        ks_iMpc = np.logspace(-2, 1, 50)  # FIXME: more samples without breaking github actions?
        ks_cont = ks_iMpc * bist.a0_Mpc
        pps_cont_t = solve_oscode(background=bist, k=ks_cont) * 1e9
        pps_cont_n = solve_oscode(background=bisn, k=ks_cont) * 1e9
        mask_t = np.isfinite(pps_cont_t)  # FIXME: manage without masking
        mask_n = np.isfinite(pps_cont_n)  # FIXME: manage without masking
        # assert np.isfinite(pps_cont_t).all()
        # assert np.isfinite(pps_cont_n).all()
        assert_allclose(pps_cont_t[mask_t], bist.P_s_approx(ks_iMpc[mask_t]) * 1e9,
                        rtol=rtol, atol=atol)
        assert_allclose(pps_cont_n[mask_n], bisn.P_s_approx(ks_iMpc[mask_n]) * 1e9,
                        rtol=rtol, atol=atol)
