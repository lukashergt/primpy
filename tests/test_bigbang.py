#!/usr/bin/env python
"""Tests for `primpy.bigbang` module."""
import pytest
from tests.test_tools import effequal
import numpy as np
from primpy.units import Mpc_m, tp_s, lp_m
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
    with pytest.raises(Exception, match="no Big Bang"):
        bb.Hubble_parameter(N=N, Omega_m0=0, Omega_K0=-0.01, h=0.7)
    with pytest.raises(Exception, match="Universe recollapses"):
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
    with pytest.raises(Exception, match="no Big Bang"):
        bb.comoving_Hubble_horizon(N=N, Omega_m0=0, Omega_K0=-0.01, h=0.7, units=units)
    with pytest.raises(Exception, match="Universe recollapses"):
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
    with pytest.raises(Exception, match="no Big Bang"):
        bb.conformal_time(N_start=100, N=200, Omega_m0=0, Omega_K0=-0.01, h=0.7)
    with pytest.raises(Exception, match="Universe recollapses"):
        bb.conformal_time(N_start=100, N=200, Omega_m0=1, Omega_K0=0.01, h=0.7)
    with pytest.raises(Exception, match="`N` needs to be either float or np.ndarray"):
        bb.conformal_time(N_start=100, N=[150, 200], Omega_m0=1, Omega_K0=0.01, h=0.7)
