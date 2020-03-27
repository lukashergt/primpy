#!/usr/bin/env python
"""Tests for `primpy.inflation` module."""
import numpy as np
from primpy.potentials import QuadraticPotential
from primpy.events import InflationEvent
from primpy.time.initialconditions import ISIC_msNO
from primpy.solver import solve


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
