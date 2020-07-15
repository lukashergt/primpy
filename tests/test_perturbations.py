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
        pert_t = CurvaturePerturbationT(background=bist, k=k, mode='scalar')
        pert_n = CurvaturePerturbationN(background=bisn, k=k, mode='scalar')
        assert pert_t.idx['Rk'] == 0
        assert pert_n.idx['Rk'] == 0
        assert pert_t.idx['dRk'] == 1
        assert pert_n.idx['dRk'] == 1
        assert pert_t.idx['steptype'] == 2
        assert pert_n.idx['steptype'] == 2
        with pytest.raises(NotImplementedError):
            pert_t(bist.x[0], bist.y[0])
        with pytest.raises(NotImplementedError):
            pert_n(bisn.x[0], bisn.y[0])
        freq_t, damp_t = pert_t.scalar_mukhanov_sasaki_frequency_damping(background=bist, k=k)
        freq_n, damp_n = pert_n.scalar_mukhanov_sasaki_frequency_damping(background=bisn, k=k)
        assert np.all(freq_t > 0)
        assert np.all(freq_n > 0)
        assert np.isfinite(damp_t).all()
        assert np.isfinite(damp_n).all()

        scalar_t, tensor_t = solve_oscode(bist, k, rtol=1e-5)
        scalar_n, tensor_n = solve_oscode(bisn, k, rtol=1e-5)
        for sol in ['one', 'two']:
            assert np.all(np.isfinite(getattr(getattr(scalar_t, sol), 't')))
            assert np.all(np.isfinite(getattr(getattr(scalar_n, sol), 'N')))
            assert np.all(np.isfinite(getattr(getattr(tensor_t, sol), 't')))
            assert np.all(np.isfinite(getattr(getattr(tensor_n, sol), 'N')))
            for var, a in itertools.product([scalar_t, scalar_n], ['Rk', 'dRk', 'steptype']):
                assert np.all(np.isfinite(getattr(getattr(var, sol), a)))
            for var, a in itertools.product([tensor_t, tensor_n], ['hk', 'dhk', 'steptype']):
                assert np.all(np.isfinite(getattr(getattr(var, sol), a)))
        assert scalar_n.P_s_RST == approx(scalar_t.P_s_RST)
        assert tensor_n.P_t_RST == approx(tensor_t.P_t_RST)


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
        P_s_disc_t, P_t_disc_t = solve_oscode(background=bist, k=ks_disc, rtol=1e-5)
        P_s_disc_n, P_t_disc_n = solve_oscode(background=bisn, k=ks_disc, rtol=1e-5)
        assert np.isfinite(P_s_disc_t).all()
        assert np.isfinite(P_s_disc_n).all()
        assert_allclose(P_s_disc_t, P_s_disc_n, rtol=rtol, atol=atol)


@pytest.mark.parametrize('K', [-1, +1])
@pytest.mark.parametrize('f_i', [10])  # FIXME: make 100, 1000 work as well
@pytest.mark.parametrize('abs_Omega_K0', [0.09, 0.009])
def test_perturbations_continuous_time_vs_efolds(K, f_i, abs_Omega_K0):
    if -K * f_i * abs_Omega_K0 >= 1:
        with pytest.raises(Exception):
            setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
    else:
        bist, bisn = setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
        rtol = 5e-3
        atol = 1e-5
        ks_iMpc = np.logspace(-4, -1, 3 * 10 + 1)  # FIXME: make this work for smaller k?
        ks_cont = ks_iMpc * bist.a0_Mpc
        P_s_cont_t, P_t_cont_t = solve_oscode(background=bist, k=ks_cont, rtol=1e-5)
        P_s_cont_n, P_t_cont_n = solve_oscode(background=bisn, k=ks_cont, rtol=1e-5)
        mask = np.isfinite(P_s_cont_t) & np.isfinite(P_s_cont_n)  # FIXME: manage without masking
        # assert np.isfinite(P_s_cont_t).all()
        # assert np.isfinite(P_s_cont_n).all()
        assert_allclose(P_s_cont_t[mask], P_s_cont_n[mask], rtol=rtol, atol=atol)


@pytest.mark.parametrize('K', [-1, +1])
@pytest.mark.parametrize('f_i', [10])  # FIXME: make 100, 1000 work as well
@pytest.mark.parametrize('abs_Omega_K0', [0.09, 0.009])
def test_perturbations_large_scales_pyoscode_vs_background(K, f_i, abs_Omega_K0):
    if -K * f_i * abs_Omega_K0 >= 1:
        with pytest.raises(Exception):
            setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
    else:
        bist, bisn = setup_background(K=K, f_i=f_i, abs_Omega_K0=abs_Omega_K0)
        rtol = 0.05
        atol = 1e-5
        ks_iMpc = np.logspace(-2, 1, 50)  # FIXME: more samples without breaking github actions?
        ks_cont = ks_iMpc * bist.a0_Mpc
        P_s_cont_t, P_t_cont_t = solve_oscode(background=bist, k=ks_cont)
        P_s_cont_n, P_t_cont_n = solve_oscode(background=bisn, k=ks_cont)
        mask_t = np.isfinite(P_s_cont_t)  # FIXME: manage without masking
        mask_n = np.isfinite(P_s_cont_n)  # FIXME: manage without masking
        # assert np.isfinite(P_s_cont_t).all()
        # assert np.isfinite(P_s_cont_n).all()
        assert_allclose(P_s_cont_t[mask_t], bist.P_s_approx(ks_iMpc[mask_t]),
                        rtol=rtol, atol=atol)
        assert_allclose(P_s_cont_n[mask_n], bisn.P_s_approx(ks_iMpc[mask_n]),
                        rtol=rtol, atol=atol)
