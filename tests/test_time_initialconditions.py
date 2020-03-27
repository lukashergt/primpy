#!/usr/bin/env python
"""Tests for `primpy.time.initialconditions` module."""
import numpy as np
from primpy.potentials import QuadraticPotential, StarobinskyPotential
from primpy.events import InflationEvent
from primpy.time.initialconditions import InflationStartIC_NiPi, ISIC_mtN, ISIC_msNO
from primpy.solver import solve


def test_InflationStartIC_NiPi():
    t_i = 1e4
    N_i = 10
    phi_i = 17
    pots = [QuadraticPotential(mass=6e-6), StarobinskyPotential(Lambda=5e-2)]
    for K in [-1, 0, +1]:
        for i, pot in enumerate(pots):
            ic = InflationStartIC_NiPi(t_i=t_i, N_i=N_i, phi_i=phi_i, K=K, potential=pot)
            y0 = np.zeros(len(ic.equations.idx))
            ic(y0)
            assert y0[0] == ic.N_i
            assert y0[1] == ic.phi_i
            assert y0[2] == -np.sqrt(pot.V(phi_i))
            assert ic.equations.K == K
            assert ic.equations.potential.V(phi_i) == pot.V(phi_i)
            assert ic.equations.potential.dV(phi_i) == pot.dV(phi_i)
            assert ic.equations.potential.d2V(phi_i) == pot.d2V(phi_i)
            assert ic.equations.potential.d3V(phi_i) == pot.d3V(phi_i)


def test_ISIC_mtN():
    t_i = 1e4
    Pot = QuadraticPotential

    mass = 6e-6
    N_tot = 60
    N_i = 10
    for K in [-1, 0, +1]:
        ic = ISIC_mtN(t_i=t_i, mass=mass, N_tot=N_tot, N_i=N_i, K=K, Potential=Pot,
                      phi_i_bracket=[15.5, 30])
        y0 = np.zeros(len(ic.equations.idx))
        ic(y0)
        assert ic.equations.potential.mass == mass
        assert y0[0] == ic.N_i
        assert ic.N_tot == N_tot

        ev = [InflationEvent(ic.equations, +1, terminal=False),
              InflationEvent(ic.equations, -1, terminal=True)]
        bist = solve(ic=ic, events=ev)
        assert np.isclose(bist.N_tot, N_tot)


def test_ISIC_msNO():
    t_i = 1e4
    Pot = QuadraticPotential
    h = 0.7

    mass = 6e-6
    N_star = 55
    N_i = 10
    for K in [-1, +1]:
        Omega_K0 = -K * 0.01
        ic = ISIC_msNO(t_i=t_i, mass=mass, N_star=N_star, N_i=N_i, Omega_K0=Omega_K0, h=h,
                       K=K, Potential=Pot, phi_i_bracket=[15.5, 30])
        y0 = np.zeros(len(ic.equations.idx))
        ic(y0)
        assert ic.equations.potential.mass == mass
        assert y0[0] == ic.N_i
        assert ic.N_star == N_star

        ev = [InflationEvent(ic.equations, +1, terminal=False),
              InflationEvent(ic.equations, -1, terminal=True)]
        bist = solve(ic=ic, events=ev)
        assert bist.N_tot > N_star
        bist.derive_approx_power(Omega_K0=Omega_K0, h=h)
        assert np.isclose(bist.N_star, N_star)
