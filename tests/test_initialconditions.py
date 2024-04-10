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
from primpy.initialconditions import SlowRollIC, InflationStartIC, ISIC_Nt, ISIC_NsOk
from primpy.solver import solve


def basic_ic_asserts(y0, ic, K, pot, N_i, Omega_Ki, phi_i, t_i):
    assert ic.N_i == N_i
    assert ic.Omega_Ki == Omega_Ki
    assert ic.phi_i == phi_i
    assert y0[0] == ic.phi_i
    if isinstance(ic.equations, InflationEquationsT):
        assert y0.size == 3
        assert ic.x_ini == t_i
        assert ic.t_i == t_i
        assert y0[1] == ic.dphidt_i
        assert y0[2] == ic.N_i
    elif isinstance(ic.equations, InflationEquationsN):
        assert y0.size == 2
        assert ic.t_i is None
        assert ic.x_ini == N_i
        assert y0[1] == ic.dphidN_i
    assert ic.equations.K == K
    assert ic.equations.potential.V(phi_i) == pot.V(phi_i)
    assert ic.equations.potential.dV(phi_i) == pot.dV(phi_i)
    assert ic.equations.potential.d2V(phi_i) == pot.d2V(phi_i)
    assert ic.equations.potential.d3V(phi_i) == pot.d3V(phi_i)


@pytest.mark.parametrize('pot', [QuadraticPotential(Lambda=0.0025),
                                 StarobinskyPotential(Lambda=5e-2)])
@pytest.mark.parametrize('K', [-1, 0, +1])
@pytest.mark.parametrize('t_i, Eq', [(1e4, InflationEquationsT), (None, InflationEquationsN)])
def test_SlowRollIC(pot, K, t_i, Eq):
    phi_i = 17

    # for N_i:
    N_i = 12
    eq = Eq(K=K, potential=pot)
    with pytest.raises(TypeError, match="Need to specify either N_i xor Omega_Ki."):
        SlowRollIC(equations=eq, phi_i=phi_i, t_i=t_i)
    ic = SlowRollIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
    y0 = np.zeros(len(ic.equations.idx))
    ic(y0)
    basic_ic_asserts(y0, ic, K, pot, N_i, ic.Omega_Ki, phi_i, t_i)

    # for Omega_Ki:
    if K != 0:
        with pytest.raises(Exception, match="Primordial curvature for open universes"):
            SlowRollIC(equations=eq, Omega_Ki=1, phi_i=phi_i, t_i=t_i)
        Omega_Ki = -K * 0.9
        ic = SlowRollIC(equations=eq, Omega_Ki=Omega_Ki, phi_i=phi_i, t_i=t_i)
        y0 = np.zeros(len(ic.equations.idx))
        ic(y0)
        basic_ic_asserts(y0, ic, K, pot, ic.N_i, Omega_Ki, phi_i, t_i)


def test_SlowRollIC_track():
    t_i = 1e4
    eta_i = 0
    phi_i = 17
    N_i = 12
    K = 0
    pot = StarobinskyPotential(Lambda=5e-2)
    eq = InflationEquationsN(K=K, potential=pot, track_time=True, track_eta=True)
    ic = SlowRollIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i, eta_i=eta_i)
    y0 = np.zeros(len(ic.equations.idx))
    ic(y0)
    assert ic.N_i == N_i
    assert ic.phi_i == phi_i
    assert ic.t_i == t_i
    assert ic.eta_i == eta_i
    assert ic.x_ini == N_i
    assert ic.equations.K == K
    assert y0.size == 4
    assert y0[0] == ic.phi_i
    assert y0[1] == ic.dphidN_i
    assert y0[2] == ic.t_i
    assert y0[3] == ic.eta_i


def test_SlowRollIC_failures():
    with pytest.raises(InflationStartError, match="V_i / 3"):
        pot = StarobinskyPotential(Lambda=5e-2)
        eq = InflationEquationsT(K=1, potential=pot)
        ic = SlowRollIC(equations=eq, N_i=0, phi_i=17)
        y0 = np.zeros(len(ic.equations.idx))
        ic(y0)
    with pytest.raises(NotImplementedError, match="`equations`"):
        pot = StarobinskyPotential(Lambda=5e-2)
        eq = InflationEquations(K=0, potential=pot)
        ic = SlowRollIC(equations=eq, N_i=0, phi_i=17)
        y0 = np.zeros(len(ic.equations.idx))
        ic(y0)


@pytest.mark.parametrize('pot', [QuadraticPotential(Lambda=0.0025),
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
    if isinstance(ic.equations, InflationEquationsT):
        assert ic.dphidt_i == -np.sqrt(ic.V_i)
    elif isinstance(ic.equations, InflationEquationsN):
        assert ic.dphidN_i == -np.sqrt(ic.V_i) / ic.H_i

    # for Omega_Ki:
    if K != 0:
        abs_Omega_Ki = 0.9
        Omega_Ki = -K * abs_Omega_Ki
        eq = Eq(K=K, potential=pot)
        ic = InflationStartIC(equations=eq, Omega_Ki=Omega_Ki, phi_i=phi_i, t_i=t_i)
        y0 = np.zeros(len(ic.equations.idx))
        ic(y0)
        basic_ic_asserts(y0, ic, K, pot, ic.N_i, Omega_Ki, phi_i, t_i)
        if isinstance(ic.equations, InflationEquationsT):
            assert ic.dphidt_i == -np.sqrt(ic.V_i)
        elif isinstance(ic.equations, InflationEquationsN):
            assert ic.dphidN_i == -np.sqrt(ic.V_i) / ic.H_i
        with pytest.raises(Exception, match="Primordial curvature for open universes"):
            InflationStartIC(equations=eq, Omega_Ki=1, phi_i=phi_i, t_i=t_i)


# noinspection DuplicatedCode
@pytest.mark.parametrize('K', [-1, 0, +1])
@pytest.mark.parametrize('t_i, Eq', [(1e4, InflationEquationsT), (None, InflationEquationsN)])
def test_ISIC_Nt_Ni(K, t_i, Eq):
    N_i = 11
    N_tot = 60
    pot = QuadraticPotential(Lambda=0.0025)
    eq = Eq(K=K, potential=pot)
    ic = ISIC_Nt(equations=eq, N_i=N_i, N_tot=N_tot, t_i=t_i, phi_i_bracket=[3, 30])
    y0 = np.zeros(len(ic.equations.idx))
    ic(y0)
    basic_ic_asserts(y0, ic, K, pot, N_i, ic.Omega_Ki, ic.phi_i, t_i)
    if isinstance(ic.equations, InflationEquationsT):
        assert ic.dphidt_i == -np.sqrt(ic.V_i)
    elif isinstance(ic.equations, InflationEquationsN):
        assert ic.dphidN_i == -np.sqrt(ic.V_i) / ic.H_i
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
    pot = QuadraticPotential(Lambda=0.0025)
    eq = Eq(K=K, potential=pot)
    if Omega_Ki >= 1:
        with pytest.raises(InflationStartError):
            ISIC_Nt(eq, Omega_Ki=Omega_Ki, N_tot=N_tot, t_i=t_i, phi_i_bracket=[3, 30])
    else:
        ic = ISIC_Nt(eq, Omega_Ki=Omega_Ki, N_tot=N_tot, t_i=t_i, phi_i_bracket=[3, 30])
        y0 = np.zeros(len(ic.equations.idx))
        ic(y0)
        basic_ic_asserts(y0, ic, K, pot, ic.N_i, Omega_Ki, ic.phi_i, t_i)
        if isinstance(ic.equations, InflationEquationsT):
            assert ic.dphidt_i == -np.sqrt(ic.V_i)
        elif isinstance(ic.equations, InflationEquationsN):
            assert ic.dphidN_i == -np.sqrt(ic.V_i) / ic.H_i
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
    pot = QuadraticPotential(Lambda=0.0025)
    N_star = 55
    h = 0.7

    # for N_i:
    N_i = 11
    Omega_K0 = -K * 0.01
    eq = Eq(K=K, potential=pot)
    ic = ISIC_NsOk(equations=eq, N_i=N_i, N_star=N_star, Omega_K0=Omega_K0, h=h, t_i=t_i,
                   phi_i_bracket=[15, 30], verbose=False)
    y0 = np.zeros(len(ic.equations.idx))
    ic(y0)
    basic_ic_asserts(y0, ic, K, pot, N_i, ic.Omega_Ki, ic.phi_i, t_i)
    if isinstance(ic.equations, InflationEquationsT):
        assert ic.dphidt_i == -np.sqrt(ic.V_i)
    elif isinstance(ic.equations, InflationEquationsN):
        assert ic.dphidN_i == -np.sqrt(ic.V_i) / ic.H_i
    assert ic.N_star == N_star
    assert ic.Omega_K0 == Omega_K0
    assert ic.h == h
    ev = [InflationEvent(ic.equations, +1, terminal=False),
          InflationEvent(ic.equations, -1, terminal=True)]
    b = solve(ic=ic, events=ev)
    b.calibrate_scale_factor(Omega_K0=Omega_K0, h=h)
    b.derive_approx_power()
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
    if isinstance(ic.equations, InflationEquationsT):
        assert ic.dphidt_i == -np.sqrt(ic.V_i)
    elif isinstance(ic.equations, InflationEquationsN):
        assert ic.dphidN_i == -np.sqrt(ic.V_i) / ic.H_i
    assert ic.N_star == N_star
    assert ic.Omega_K0 == Omega_K0
    assert ic.h == h
    ev = [InflationEvent(ic.equations, +1, terminal=False),
          InflationEvent(ic.equations, -1, terminal=True)]
    b = solve(ic=ic, events=ev)
    b.calibrate_scale_factor(Omega_K0=Omega_K0, h=h)
    b.derive_approx_power()
    assert b.N_tot > N_star
    assert pytest.approx(b.N_star) == N_star
