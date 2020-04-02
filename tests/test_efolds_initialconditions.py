#!/usr/bin/env python
"""Tests for `primpy.efolds.initialconditions` module."""
import pytest
import numpy as np
from primpy.potentials import QuadraticPotential, StarobinskyPotential
from primpy.events import InflationEvent
from primpy.efolds.initialconditions import InflationStartIC_NiPi, ISIC_mtN, ISIC_msNO
from primpy.solver import solve


def test_InflationStartIC_NiPi():
    N_i = 10
    phi_i = 17
    pots = [QuadraticPotential(mass=6e-6), StarobinskyPotential(Lambda=5e-2)]
    for K in [-1, 0, +1]:
        for i, pot in enumerate(pots):
            ic = InflationStartIC_NiPi(N_i=N_i, phi_i=phi_i, K=K, potential=pot)

            assert ic.x_ini == N_i
            assert ic.x_end == 1e300
            assert ic.phi_i == phi_i
            assert ic.H_i == np.sqrt(pot.V(phi_i) / 2 - K * np.exp(-2 * N_i))
            assert ic.dphidN_i == -np.sqrt(pot.V(phi_i)) / ic.H_i

            assert len(ic.equations.idx) == 2
            assert ic.equations.idx['phi'] == 0
            assert ic.equations.idx['dphidN'] == 1
            assert ic.equations.K == K
            assert ic.equations.potential == pot

            y0 = np.zeros(len(ic.equations.idx))
            ic(y0)
            assert y0[0] == ic.phi_i
            assert y0[1] == ic.dphidN_i


def test_ISIC_mtN():
    Pot = QuadraticPotential

    mass = 6e-6
    N_tot = 60
    N_i = 10
    for K in [-1, 0, +1]:
        ic = ISIC_mtN(mass=mass, N_tot=N_tot, N_i=N_i, K=K, Potential=Pot,
                      phi_i_bracket=[15.5, 30])

        assert len(ic.equations.idx) == 2
        assert ic.equations.idx['phi'] == 0
        assert ic.equations.idx['dphidN'] == 1
        assert ic.equations.potential.mass == mass

        y0 = np.zeros(len(ic.equations.idx))
        ic(y0)
        assert y0[0] == ic.phi_i
        assert y0[1] == ic.dphidN_i
        assert ic.N_tot == N_tot

        ev = [InflationEvent(ic.equations, +1, terminal=False),
              InflationEvent(ic.equations, -1, terminal=True)]
        bisn = solve(ic=ic, events=ev, rtol=1e-10, atol=1e-10)
        assert bisn.N_tot == pytest.approx(N_tot, rel=1e-5, abs=1e-5)


def test_ISIC_msNO():
    Pot = QuadraticPotential
    h = 0.7

    mass = 6e-6
    N_star = 55
    N_i = 10
    for K in [-1, +1]:
        Omega_K0 = -K * 0.01
        ic = ISIC_msNO(mass=mass, N_star=N_star, N_i=N_i, Omega_K0=Omega_K0, h=h,
                       K=K, Potential=Pot, phi_i_bracket=[15.5, 30])

        assert len(ic.equations.idx) == 2
        assert ic.equations.idx['phi'] == 0
        assert ic.equations.idx['dphidN'] == 1
        assert ic.equations.potential.mass == mass

        y0 = np.zeros(len(ic.equations.idx))
        ic(y0)
        assert y0[0] == ic.phi_i
        assert y0[1] == ic.dphidN_i
        assert ic.N_star == N_star

        ev = [InflationEvent(ic.equations, +1, terminal=False),
              InflationEvent(ic.equations, -1, terminal=True)]
        bisn = solve(ic=ic, events=ev, rtol=1e-10, atol=1e-10)
        assert bisn.N_tot > N_star
        bisn.derive_approx_power(Omega_K0=Omega_K0, h=h)
        assert bisn.N_star == pytest.approx(N_star, rel=1e-5, abs=1e-5)
