#!/usr/bin/env python
"""Tests for `primpy.bigbang` module."""
import pytest
from tests.test_tools import effequal
import numpy as np
from primpy.exceptionhandling import BigBangWarning, BigBangError
from primpy.units import Mpc_m, tp_s, lp_m
from primpy.potentials import QuadraticPotential
from primpy.time.inflation import InflationEquationsT
from primpy.initialconditions import InflationStartIC, ISIC_Nt
from primpy.events import InflationEvent, UntilNEvent
from primpy.solver import solve
import primpy.bigbang as bb


def test_not_implemented_units():
    with pytest.raises(NotImplementedError):
        bb.get_H0(h=0.7, units='Mpc')
    with pytest.raises(NotImplementedError):
        bb.get_a0(h=0.7, Omega_K0=-0.01, units='H0')


@pytest.mark.parametrize('h', [0.3, 0.5, 0.7, 0.9])
def test_get_H0(h):
    """Tests for `get_H0`."""
    H0_km_per_Mpc_per_s = bb.get_H0(h=h, units='H0')
    H0_per_s = bb.get_H0(h=h, units='SI')
    H0_per_tp = bb.get_H0(h=h, units='planck')
    assert effequal(H0_per_s) == H0_km_per_Mpc_per_s * 1e3 / Mpc_m
    assert effequal(H0_per_s) == H0_per_tp / tp_s
    assert effequal(H0_per_tp / tp_s) == H0_km_per_Mpc_per_s * 1e3 / Mpc_m


@pytest.mark.parametrize('h', [0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize('Omega_K0', [-0.15, -0.01, 0.01, 0.15])
def test_get_a0(h, Omega_K0):
    """Tests for `get_a0`."""
    a0_Mpc = bb.get_a0(h=h, Omega_K0=Omega_K0, units='Mpc')
    a0__lp = bb.get_a0(h=h, Omega_K0=Omega_K0, units='planck')
    a0___m = bb.get_a0(h=h, Omega_K0=Omega_K0, units='SI')
    assert effequal(a0___m) == a0_Mpc * Mpc_m
    assert effequal(a0___m) == a0__lp * lp_m
    assert effequal(a0__lp * lp_m) == a0_Mpc * Mpc_m


@pytest.mark.parametrize('units', ['Mpc', 'planck', 'SI'])
def test_get_a0_flat(units):
    assert 1 == bb.get_a0(h=0.7, Omega_K0=0, units=units)


@pytest.mark.parametrize('h', [0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize('Omega_K0', [-0.15, -0.01, 0.01, 0.15])
def test_get_N_BBN(h, Omega_K0):
    N0 = np.log(bb.get_a0(h=h, Omega_K0=Omega_K0, units='planck'))
    N_BBN = bb.get_N_BBN(h=h, Omega_K0=Omega_K0)
    assert 100 < N_BBN < N0


@pytest.mark.parametrize('N_tot', [10, 20, 60, 70, 90])
def test_get_w_reh(N_tot):
    h = 0.7
    Omega_K0 = -0.01
    K = -np.sign(Omega_K0)
    Omega_Ki = 10 * Omega_K0
    pot = QuadraticPotential(Lambda=np.sqrt(6e-6))
    t_i = 7e4
    eq = InflationEquationsT(K=K, potential=pot)
    ic = ISIC_Nt(eq, N_tot=N_tot, Omega_Ki=Omega_Ki, phi_i_bracket=(5, 30), t_i=t_i)
    ev = [InflationEvent(eq, +1, terminal=False),
          InflationEvent(eq, -1, terminal=True)]
    bist = solve(ic=ic, events=ev)
    bist.calibrate_scale_factor(Omega_K0=Omega_K0, h=h)

    N_BBN = bb.get_N_BBN(h=h, Omega_K0=Omega_K0)
    cHH_BBN_Mpc = bb.comoving_Hubble_horizon(N=N_BBN, Omega_m0=0.3, Omega_K0=Omega_K0, h=h,
                                             units='Mpc')

    w_reh, delta_reh = bb.get_w_delta_reh(N_end=bist.N_end,
                                          N_reh=N_BBN,
                                          log_cHH_end=np.log(bist.cHH_end_Mpc),
                                          log_cHH_reh=np.log(cHH_BBN_Mpc))
    if N_tot < 15:
        assert w_reh < -1 / 3
    elif N_tot < 25:
        assert -1 / 3 < w_reh < 0
    elif N_tot < 65:
        assert 0 < w_reh < 1 / 3
    elif N_tot < 80:
        assert 1 / 3 < w_reh < 1
    else:
        assert 1 < w_reh


@pytest.mark.parametrize('h', [0.3, 0.5, 0.7, 0.9])
def test_Omega_r0(h):
    assert 0 < bb.get_Omega_r0(h) < 1e-4 / h**2


@pytest.mark.parametrize('h', [0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize('Omega_K0', [-0.15, -0.01, 0.01, 0.15])
def test_Hubble_parameter(h, Omega_K0):
    N = np.linspace(0, 200, 201)
    bb.Hubble_parameter(N=N, Omega_m0=0.3, Omega_K0=Omega_K0, h=h)


def test_no_Big_Bang_line():
    assert 1 == bb.no_Big_Bang_line(Omega_m0=0)
    assert 2 == bb.no_Big_Bang_line(Omega_m0=0.5)
    bb.no_Big_Bang_line(Omega_m0=1)
    with pytest.raises(ValueError, match="Matter density can't be negative"):
        bb.no_Big_Bang_line(Omega_m0=-1)


def test_expand_recollapse_line():
    assert 0 == bb.expand_recollapse_line(Omega_m0=0)
    assert 0 == bb.expand_recollapse_line(Omega_m0=0.5)
    assert effequal(0) == bb.expand_recollapse_line(Omega_m0=1)
    with pytest.raises(ValueError, match="Matter density can't be negative"):
        bb.expand_recollapse_line(Omega_m0=-1)


def test_Hubble_parameter_exceptions():
    N = np.linspace(0, 200, 201)
    with pytest.raises(BigBangError, match="no Big Bang"):
        bb.Hubble_parameter(N=N, Omega_m0=0, Omega_K0=-0.01, h=0.7)
    with pytest.warns(BigBangWarning, match="Universe recollapses"):
        bb.Hubble_parameter(N=N, Omega_m0=1, Omega_K0=0.01, h=0.7)


@pytest.mark.parametrize('h', [0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize('Omega_K0', [-0.15, -0.01, 0.01, 0.15])
@pytest.mark.parametrize('units', ['planck', 'Mpc', 'SI'])
def test_comoving_Hubble_horizon(h, Omega_K0, units):
    N = np.linspace(0, 200, 201)
    bb.comoving_Hubble_horizon(N=N, Omega_m0=0.3, Omega_K0=Omega_K0, h=h, units=units)


@pytest.mark.parametrize('units', ['planck', 'Mpc', 'SI'])
def test_comoving_Hubble_horizon_exceptions(units):
    N = np.linspace(0, 200, 201)
    with pytest.raises(BigBangError, match="no Big Bang"):
        bb.comoving_Hubble_horizon(N=N, Omega_m0=0, Omega_K0=-0.01, h=0.7, units=units)
    with pytest.warns(BigBangWarning, match="Universe recollapses"):
        bb.comoving_Hubble_horizon(N=N, Omega_m0=1, Omega_K0=0.01, h=0.7, units=units)


@pytest.mark.parametrize('h', [0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize('Omega_K0', [-0.15, -0.01, 0.01, 0.15])
@pytest.mark.parametrize('N_BB', [60, 100])
def test_conformal_time(h, Omega_K0, N_BB):
    eta1 = bb.conformal_time(N_start=N_BB, N=200, Omega_m0=0.3, Omega_K0=Omega_K0, h=h)[0]
    eta2 = bb.conformal_time(N_start=N_BB, N=250, Omega_m0=0.3, Omega_K0=Omega_K0, h=h)[0]
    assert eta1 == pytest.approx(eta2)

    N = np.linspace(N_BB, 200, (200-N_BB)//10+1)
    etas = bb.conformal_time(N_start=N_BB, N=N, Omega_m0=0.3, Omega_K0=Omega_K0, h=h)
    assert etas[-2] == pytest.approx(etas[-1])


def test_conformal_time_exceptions():
    with pytest.raises(BigBangError, match="no Big Bang"):
        bb.conformal_time(N_start=100, N=200, Omega_m0=0, Omega_K0=-0.01, h=0.7)
    with pytest.warns(BigBangWarning, match="Universe recollapses"):
        bb.conformal_time(N_start=100, N=200, Omega_m0=1, Omega_K0=0.01, h=0.7)
    with pytest.raises(TypeError, match="`N` needs to be either float or np.ndarray"):
        bb.conformal_time(N_start=100, N=[150, 200], Omega_m0=1, Omega_K0=0.01, h=0.7)


@pytest.mark.parametrize('Omega_m0', [0.2, 0.4])
@pytest.mark.parametrize('h', [0.5, 0.7])
@pytest.mark.parametrize('Omega_K0', [-0.15, -0.01, 0.01, 0.09])
@pytest.mark.parametrize('f_i', [1, 10])
def test_conformal_time_ratio(f_i, Omega_K0, h, Omega_m0):
    K = -np.sign(Omega_K0)
    Omega_Ki = f_i * Omega_K0
    pot = QuadraticPotential(Lambda=np.sqrt(6e-6))
    t_i = 7e4
    phi_i = 17
    eta_i = 0

    eq = InflationEquationsT(K=K, potential=pot, track_eta=True)
    ic_f = InflationStartIC(eq, Omega_Ki=Omega_Ki, phi_i=phi_i, t_i=t_i, eta_i=eta_i)
    ic_b = InflationStartIC(eq, Omega_Ki=Omega_Ki, phi_i=phi_i, t_i=t_i, eta_i=eta_i, x_end=1)
    ev = [InflationEvent(eq, +1, terminal=False),
          InflationEvent(eq, -1, terminal=True)]
    bist_f = solve(ic=ic_f, events=ev)
    bist_f.calibrate_scale_factor(Omega_K0=Omega_K0, h=h)
    bist_b = solve(ic=ic_b, events=[UntilNEvent(eq, 0)])
    assert np.isclose(bist_b.eta[-1], bist_b.eta[-2])

    ratio = bb.conformal_time_ratio(Omega_m0=Omega_m0, Omega_K0=Omega_K0, h=h,
                                    b_forward=bist_f, b_backward=bist_b)
    if np.log10(f_i) < 0.5:
        assert ratio < 1
    else:
        assert ratio > 1

    ratio2 = bb.conformal_time_ratio(Omega_m0=Omega_m0, Omega_K0=Omega_K0, h=h, b_forward=bist_f)
    assert ratio2 < ratio
