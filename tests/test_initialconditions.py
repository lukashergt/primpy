#!/usr/bin/env python
"""Tests for `primpy.initialconditions` module."""
import pytest
import numpy as np
from primpy.potentials import QuadraticPotential, StarobinskyPotential
from primpy.events import InflationEvent
from primpy.time.inflation import InflationEquationsT
from primpy.efolds.inflation import InflationEquationsN
from primpy.initialconditions import InflationStartIC, ISIC_Nt, ISIC_NsOk
from primpy.solver import solve


def basic_ic_asserts(y0, ic, K, pot, N_i, phi_i, t_i):
    assert ic.N_i == N_i
    assert ic.phi_i == phi_i
    assert ic.eta_i is None
    assert y0[0] == ic.phi_i
    if isinstance(ic.equations, InflationEquationsT):
        assert y0.size == 3
        assert ic.x_ini == t_i
        assert ic.t_i == t_i
        assert ic.dphidt_i == -np.sqrt(ic.V_i)
        assert y0[1] == ic.dphidt_i
        assert y0[2] == ic.N_i
    elif isinstance(ic.equations, InflationEquationsN):
        assert y0.size == 2
        assert ic.t_i is None
        assert ic.x_ini == N_i
        assert ic.dphidN_i == -np.sqrt(ic.V_i) / ic.H_i
        assert y0[1] == ic.dphidN_i
    assert ic.equations.K == K
    assert ic.equations.potential.V(phi_i) == pot.V(phi_i)
    assert ic.equations.potential.dV(phi_i) == pot.dV(phi_i)
    assert ic.equations.potential.d2V(phi_i) == pot.d2V(phi_i)
    assert ic.equations.potential.d3V(phi_i) == pot.d3V(phi_i)


def test_InflationStartIC_NiPi():
    pots = [QuadraticPotential(mass=6e-6), StarobinskyPotential(Lambda=5e-2)]
    N_i = 10
    phi_i = 17
    for t_i, InflationEquations in zip([1e4, None], [InflationEquationsT, InflationEquationsN]):
        for K in [-1, 0, +1]:
            for i, pot in enumerate(pots):
                eq = InflationEquations(K=K, potential=pot)
                ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
                y0 = np.zeros(len(ic.equations.idx))
                ic(y0)
                basic_ic_asserts(y0, ic, K, pot, N_i, phi_i, t_i)


def test_ISIC_NiNt():
    pot = QuadraticPotential(mass=6e-6)
    N_i = 10
    N_tot = 60
    for t_i, InflationEquations in zip([1e4, None], [InflationEquationsT, InflationEquationsN]):
        for K in [-1, 0, +1]:
            eq = InflationEquations(K=K, potential=pot)
            ic = ISIC_Nt(equations=eq, N_i=N_i, N_tot=N_tot, t_i=t_i, phi_i_bracket=[15.5, 30])
            y0 = np.zeros(len(ic.equations.idx))
            ic(y0)
            basic_ic_asserts(y0, ic, K, pot, N_i, ic.phi_i, t_i)
            assert ic.N_tot == N_tot
            ev = [InflationEvent(ic.equations, +1, terminal=False),
                  InflationEvent(ic.equations, -1, terminal=True)]
            if isinstance(eq, InflationEquationsT):
                bist = solve(ic=ic, events=ev)
                assert pytest.approx(bist.N_tot) == N_tot
            elif isinstance(eq, InflationEquationsN):
                bisn = solve(ic=ic, events=ev, rtol=1e-10, atol=1e-10)
                assert pytest.approx(bisn.N_tot, rel=1e-6, abs=1e-6) == N_tot


def test_ISIC_NiNsOk():
    pot = QuadraticPotential(mass=6e-6)
    N_i = 10
    N_star = 55
    h = 0.7
    for t_i, InflationEquations in zip([1e4, None], [InflationEquationsT, InflationEquationsN]):
        for K in [-1, +1]:
            Omega_K0 = -K * 0.01
            eq = InflationEquations(K=K, potential=pot)
            ic = ISIC_NsOk(equations=eq, N_i=N_i, N_star=N_star, Omega_K0=Omega_K0, h=h,
                           t_i=t_i, phi_i_bracket=[15.5, 30])
            y0 = np.zeros(len(ic.equations.idx))
            ic(y0)
            basic_ic_asserts(y0, ic, K, pot, N_i, ic.phi_i, t_i)
            assert ic.N_star == N_star
            assert ic.Omega_K0 == Omega_K0
            assert ic.h == h
            ev = [InflationEvent(ic.equations, +1, terminal=False),
                  InflationEvent(ic.equations, -1, terminal=True)]
            if isinstance(eq, InflationEquationsT):
                bist = solve(ic=ic, events=ev)
                assert bist.N_tot > N_star
                bist.derive_approx_power(Omega_K0=Omega_K0, h=h)
                assert pytest.approx(bist.N_star) == N_star
            elif isinstance(eq, InflationEquationsN):
                bisn = solve(ic=ic, events=ev, rtol=1e-10, atol=1e-10)
                assert bisn.N_tot > N_star
                bisn.derive_approx_power(Omega_K0=Omega_K0, h=h)
                assert pytest.approx(bisn.N_star, rel=1e-6, abs=1e-6) == N_star
