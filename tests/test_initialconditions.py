#!/usr/bin/env python
"""Tests for `primpy.initialconditions` module."""
import pytest
import numpy as np
from primpy.exceptionhandling import InflationStartError
from primpy.potentials import QuadraticPotential, StarobinskyPotential
from primpy.events import InflationEvent
from primpy.inflation import InflationEquations
from primpy.time.inflation import InflationEquationsT
from primpy.efolds.inflation import InflationEquationsN
from primpy.initialconditions import InflationStartIC, ISIC_Nt, ISIC_NsOk
from primpy.solver import solve


def basic_ic_asserts(y0, ic, K, pot, N_i, Omega_Ki, phi_i, t_i):
    assert ic.N_i == N_i
    assert ic.Omega_Ki == Omega_Ki
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


@pytest.mark.parametrize('pot', [QuadraticPotential(Lambda=np.sqrt(6e-6)),
                                 StarobinskyPotential(Lambda=5e-2)])
@pytest.mark.parametrize('K', [-1, 0, +1])
@pytest.mark.parametrize('t_i, Eq', [(1e4, InflationEquationsT), (None, InflationEquationsN)])
def test_InflationStartIC(pot, K, t_i, Eq):
    phi_i = 17

    # for N_i:
    N_i = 10
    with pytest.raises(NotImplementedError):
        eq = InflationEquations(K=K, potential=pot)
        ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
        y0 = np.zeros(len(ic.equations.idx))
        ic(y0)
    eq = Eq(K=K, potential=pot)
    with pytest.raises(TypeError, match="Need to specify either N_i or Omega_Ki."):
        InflationStartIC(equations=eq, phi_i=phi_i, t_i=t_i)
    ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
    y0 = np.zeros(len(ic.equations.idx))
    ic(y0)
    basic_ic_asserts(y0, ic, K, pot, N_i, ic.Omega_Ki, phi_i, t_i)

    # for Omega_Ki:
    if K != 0:
        abs_Omega_Ki = 0.9
        Omega_Ki = -K * abs_Omega_Ki
        eq = Eq(K=K, potential=pot)
        ic = InflationStartIC(equations=eq, Omega_Ki=Omega_Ki, phi_i=phi_i, t_i=t_i)
        y0 = np.zeros(len(ic.equations.idx))
        ic(y0)
        basic_ic_asserts(y0, ic, K, pot, ic.N_i, Omega_Ki, phi_i, t_i)
        with pytest.raises(Exception, match="Primordial curvature for open universes"):
            InflationStartIC(equations=eq, Omega_Ki=1, phi_i=phi_i, t_i=t_i)


# noinspection DuplicatedCode
@pytest.mark.parametrize('K', [-1, 0, +1])
@pytest.mark.parametrize('t_i, Eq', [(1e4, InflationEquationsT), (None, InflationEquationsN)])
def test_ISIC_Nt_Ni(K, t_i, Eq):
    N_i = 11
    N_tot = 60
    pot = QuadraticPotential(Lambda=np.sqrt(6e-6))
    eq = Eq(K=K, potential=pot)
    ic = ISIC_Nt(equations=eq, N_i=N_i, N_tot=N_tot, t_i=t_i, phi_i_bracket=[3, 30])
    y0 = np.zeros(len(ic.equations.idx))
    ic(y0)
    basic_ic_asserts(y0, ic, K, pot, N_i, ic.Omega_Ki, ic.phi_i, t_i)
    assert ic.N_tot == N_tot
    ev = [InflationEvent(ic.equations, +1, terminal=False),
          InflationEvent(ic.equations, -1, terminal=True)]
    if isinstance(eq, InflationEquationsT):
        bist = solve(ic=ic, events=ev)
        assert pytest.approx(bist.N_tot) == N_tot
    elif isinstance(eq, InflationEquationsN):
        bisn = solve(ic=ic, events=ev, rtol=1e-10, atol=1e-10)
        assert pytest.approx(bisn.N_tot, rel=1e-6, abs=1e-6) == N_tot


# noinspection DuplicatedCode
@pytest.mark.parametrize('K', [-1, +1])
@pytest.mark.parametrize('abs_Omega_Ki', [0.9, 10])
@pytest.mark.parametrize('t_i, Eq', [(1e4, InflationEquationsT), (None, InflationEquationsN)])
def test_ISIC_Nt_Oi(K, abs_Omega_Ki, t_i, Eq):
    Omega_Ki = -K * abs_Omega_Ki
    N_tot = 60
    pot = QuadraticPotential(Lambda=np.sqrt(6e-6))
    eq = Eq(K=K, potential=pot)
    if Omega_Ki >= 1:
        with pytest.raises(InflationStartError):
            ISIC_Nt(eq, Omega_Ki=Omega_Ki, N_tot=N_tot, t_i=t_i, phi_i_bracket=[3, 30])
    else:
        ic = ISIC_Nt(eq, Omega_Ki=Omega_Ki, N_tot=N_tot, t_i=t_i, phi_i_bracket=[3, 30])
        y0 = np.zeros(len(ic.equations.idx))
        ic(y0)
        basic_ic_asserts(y0, ic, K, pot, ic.N_i, Omega_Ki, ic.phi_i, t_i)
        assert ic.N_tot == N_tot
        ev = [InflationEvent(ic.equations, +1, terminal=False),
              InflationEvent(ic.equations, -1, terminal=True)]
        if isinstance(eq, InflationEquationsT):
            bist = solve(ic=ic, events=ev)
            assert pytest.approx(bist.N_tot) == N_tot
        elif isinstance(eq, InflationEquationsN):
            bisn = solve(ic=ic, events=ev, rtol=1e-10, atol=1e-10)
            assert pytest.approx(bisn.N_tot, rel=1e-6, abs=1e-6) == N_tot


# noinspection DuplicatedCode
@pytest.mark.parametrize('K', [-1, +1])
@pytest.mark.parametrize('t_i, Eq', [(1e4, InflationEquationsT), (None, InflationEquationsN)])
def test_ISIC_NsOk(K, t_i, Eq):
    pot = QuadraticPotential(Lambda=np.sqrt(6e-6))
    N_star = 55
    h = 0.7

    # for N_i:
    N_i = 11
    Omega_K0 = -K * 0.01
    eq = Eq(K=K, potential=pot)
    ic = ISIC_NsOk(equations=eq, N_i=N_i, N_star=N_star, Omega_K0=Omega_K0, h=h, t_i=t_i,
                   phi_i_bracket=[15, 30], verbose=True)
    y0 = np.zeros(len(ic.equations.idx))
    ic(y0)
    basic_ic_asserts(y0, ic, K, pot, N_i, ic.Omega_Ki, ic.phi_i, t_i)
    assert ic.N_star == N_star
    assert ic.Omega_K0 == Omega_K0
    assert ic.h == h
    ev = [InflationEvent(ic.equations, +1, terminal=False),
          InflationEvent(ic.equations, -1, terminal=True)]
    b = solve(ic=ic, events=ev)
    b.derive_approx_power(Omega_K0=Omega_K0, h=h)
    assert b.N_tot > N_star
    assert pytest.approx(b.N_star) == N_star

    # for Omega_Ki:
    abs_Omega_Ki = 0.9
    Omega_Ki = -K * abs_Omega_Ki
    Omega_K0 = -K * 0.01
    eq = Eq(K=K, potential=pot)
    ic = ISIC_NsOk(equations=eq, Omega_Ki=Omega_Ki, N_star=N_star, Omega_K0=Omega_K0, h=h, t_i=t_i,
                   phi_i_bracket=[15, 30])
    y0 = np.zeros(len(ic.equations.idx))
    ic(y0)
    basic_ic_asserts(y0, ic, K, pot, ic.N_i, Omega_Ki, ic.phi_i, t_i)
    assert ic.N_star == N_star
    assert ic.Omega_K0 == Omega_K0
    assert ic.h == h
    ev = [InflationEvent(ic.equations, +1, terminal=False),
          InflationEvent(ic.equations, -1, terminal=True)]
    b = solve(ic=ic, events=ev)
    b.derive_approx_power(Omega_K0=Omega_K0, h=h)
    assert b.N_tot > N_star
    assert pytest.approx(b.N_star) == N_star
