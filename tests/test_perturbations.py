#!/usr/bin/env python
"""Tests for `primpy.perturbation` module."""
import pytest
from pytest import approx
import itertools
import numpy as np
from numpy.testing import assert_allclose
from primpy.potentials import QuadraticPotential
from primpy.events import InflationEvent, CollapseEvent
from primpy.time.inflation import InflationEquationsT
from primpy.efolds.inflation import InflationEquationsN
from primpy.initialconditions import InflationStartIC
from primpy.time.perturbations import PerturbationT
from primpy.efolds.perturbations import PerturbationN
from primpy.solver import solve
from primpy.oscode_solver import solve_oscode


def setup_background(K, f_i, abs_Omega_K0):
    pot = QuadraticPotential(mass=6e-6)
    phi_i = 16
    Omega_K0 = -K * abs_Omega_K0
    Omega_Ki = f_i * Omega_K0
    h = 0.7

    eq_t = InflationEquationsT(K=K, potential=pot)
    eq_n = InflationEquationsN(K=K, potential=pot)
    t_eval = np.logspace(np.log10(5e4), np.log10(5e6), int(2e4))
    ic_t = InflationStartIC(eq_t, phi_i=phi_i, Omega_Ki=Omega_Ki, t_i=t_eval[0])
    ic_n = InflationStartIC(eq_n, phi_i=phi_i, Omega_Ki=Omega_Ki, t_i=None)
    N_eval = np.linspace(ic_n.N_i, 100, int(1e5))
    ev_t = [InflationEvent(eq_t, +1, terminal=False),
            InflationEvent(eq_t, -1, terminal=True),
            CollapseEvent(eq_t)]
    ev_n = [InflationEvent(eq_n, +1, terminal=False),
            InflationEvent(eq_n, -1, terminal=True),
            CollapseEvent(eq_n)]
    bist = solve(ic=ic_t, events=ev_t, t_eval=t_eval)
    bisn = solve(ic=ic_n, events=ev_n, t_eval=N_eval)
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
        pert_t = PerturbationT(background=bist, k=k)
        pert_n = PerturbationN(background=bisn, k=k)
        assert pert_t.scalar.idx['Rk'] == 0
        assert pert_n.scalar.idx['Rk'] == 0
        assert pert_t.scalar.idx['dRk'] == 1
        assert pert_n.scalar.idx['dRk'] == 1
        assert pert_t.tensor.idx['hk'] == 0
        assert pert_n.tensor.idx['hk'] == 0
        assert pert_t.tensor.idx['dhk'] == 1
        assert pert_n.tensor.idx['dhk'] == 1
        with pytest.raises(NotImplementedError):
            pert_t.scalar(bist.x[0], bist.y[0])
        with pytest.raises(NotImplementedError):
            pert_t.tensor(bist.x[0], bist.y[0])
        with pytest.raises(NotImplementedError):
            pert_n.scalar(bisn.x[0], bisn.y[0])
        with pytest.raises(NotImplementedError):
            pert_n.tensor(bisn.x[0], bisn.y[0])
        freq_t, damp_t = pert_t.scalar.mukhanov_sasaki_frequency_damping()
        freq_n, damp_n = pert_n.scalar.mukhanov_sasaki_frequency_damping()
        assert np.all(freq_t > 0)
        assert np.all(freq_n > 0)
        assert np.isfinite(damp_t).all()
        assert np.isfinite(damp_n).all()
        freq_t, damp_t = pert_t.tensor.mukhanov_sasaki_frequency_damping()
        freq_n, damp_n = pert_n.tensor.mukhanov_sasaki_frequency_damping()
        assert np.all(freq_t > 0)
        assert np.all(freq_n > 0)
        assert np.isfinite(damp_t).all()
        assert np.isfinite(damp_n).all()

        pert_t = solve_oscode(background=bist, k=k, rtol=1e-5)
        pert_n = solve_oscode(background=bisn, k=k, rtol=1e-5, even_grid=True)
        for sol in ['one', 'two']:
            assert np.all(np.isfinite(getattr(getattr(pert_t.scalar, sol), 't')))
            assert np.all(np.isfinite(getattr(getattr(pert_n.scalar, sol), 'N')))
            assert np.all(np.isfinite(getattr(getattr(pert_t.tensor, sol), 't')))
            assert np.all(np.isfinite(getattr(getattr(pert_n.tensor, sol), 'N')))
            for scalar, a in itertools.product([pert_t.scalar, pert_n.scalar],
                                               ['Rk', 'dRk', 'steptype']):
                assert np.all(np.isfinite(getattr(getattr(scalar, sol), a)))
            for tensor, a in itertools.product([pert_t.tensor, pert_n.tensor],
                                               ['hk', 'dhk', 'steptype']):
                assert np.all(np.isfinite(getattr(getattr(tensor, sol), a)))
        assert pert_n.scalar.P_s_RST == approx(pert_t.scalar.P_s_RST)
        assert pert_n.tensor.P_t_RST == approx(pert_t.tensor.P_t_RST)


@pytest.mark.parametrize('K', [-1, +1])
@pytest.mark.parametrize('f_i', [10])  # FIXME: make 100, 1000 work as well
@pytest.mark.parametrize('abs_Omega_K0', [0.09, 0.009])
def test_perturbations_discrete_time_efolds(K, f_i, abs_Omega_K0):
    if -K * f_i * abs_Omega_K0 >= 1:
        with pytest.raises(Exception):
            setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
    else:
        bist, bisn = setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
        ks_disc = np.arange(1, 100, 1)
        pps_t = solve_oscode(background=bist, k=ks_disc, rtol=1e-5)
        pps_n = solve_oscode(background=bisn, k=ks_disc, rtol=1e-5, even_grid=True)
        assert np.isfinite(pps_t.P_s_RST).all()
        assert np.isfinite(pps_t.P_t_RST).all()
        assert np.isfinite(pps_n.P_s_RST).all()
        assert np.isfinite(pps_n.P_t_RST).all()
        assert_allclose(pps_t.P_s_RST * 1e9, pps_n.P_s_RST * 1e9, rtol=5e-4, atol=1e-6)
        assert_allclose(pps_t.P_t_RST * 1e9, pps_n.P_t_RST * 1e9, rtol=5e-4, atol=1e-6)


@pytest.mark.parametrize('K', [-1, +1])
@pytest.mark.parametrize('f_i', [10])  # FIXME: make 100, 1000 work as well
@pytest.mark.parametrize('abs_Omega_K0', [0.09, 0.009])
def test_perturbations_continuous_time_vs_efolds(K, f_i, abs_Omega_K0):
    if -K * f_i * abs_Omega_K0 >= 1:
        with pytest.raises(Exception):
            setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
    else:
        bist, bisn = setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
        ks_iMpc = np.logspace(-4, 0, 4 * 10 + 1)
        ks_cont = ks_iMpc * bist.a0_Mpc
        pps_t = solve_oscode(background=bist, k=ks_cont, rtol=1e-5)
        pps_n = solve_oscode(background=bisn, k=ks_cont, rtol=1e-5, even_grid=True)
        assert np.isfinite(pps_t.P_s_RST).all()
        assert np.isfinite(pps_t.P_t_RST).all()
        assert np.isfinite(pps_n.P_s_RST).all()
        assert np.isfinite(pps_n.P_t_RST).all()
        assert_allclose(pps_t.P_s_RST * 1e9, pps_n.P_s_RST * 1e9, rtol=1e-3, atol=1e-6)
        assert_allclose(pps_t.P_t_RST * 1e9, pps_n.P_t_RST * 1e9, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize('K', [-1, +1])
@pytest.mark.parametrize('f_i', [10])  # FIXME: make 100, 1000 work as well
@pytest.mark.parametrize('abs_Omega_K0', [0.09, 0.009])
def test_perturbations_large_scales_pyoscode_vs_background(K, f_i, abs_Omega_K0):
    if -K * f_i * abs_Omega_K0 >= 1:
        with pytest.raises(Exception):
            setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
    else:
        bist, bisn = setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
        ks_iMpc = np.logspace(-1, 1, 100)
        ks_cont = ks_iMpc * bist.a0_Mpc
        pps_t = solve_oscode(background=bist, k=ks_cont)
        pps_n = solve_oscode(background=bisn, k=ks_cont, even_grid=True)
        assert np.isfinite(pps_t.P_s_RST).all()
        assert np.isfinite(pps_t.P_t_RST).all()
        assert np.isfinite(pps_n.P_s_RST).all()
        assert np.isfinite(pps_n.P_t_RST).all()
        assert_allclose(pps_t.P_s_RST * 1e9, bist.P_s_approx(ks_iMpc) * 1e9, rtol=0.02, atol=1e-6)
        assert_allclose(pps_t.P_t_RST * 1e9, bist.P_t_approx(ks_iMpc) * 1e9, rtol=0.02, atol=1e-6)
        assert_allclose(pps_n.P_s_RST * 1e9, bisn.P_s_approx(ks_iMpc) * 1e9, rtol=0.02, atol=1e-6)
        assert_allclose(pps_n.P_t_RST * 1e9, bisn.P_t_approx(ks_iMpc) * 1e9, rtol=0.02, atol=1e-6)
