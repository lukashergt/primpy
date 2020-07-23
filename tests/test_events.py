#!/usr/bin/env python
"""Tests for `primpy.inflation` module."""
import pytest
from pytest import approx
import numpy as np
from primpy.potentials import QuadraticPotential
from primpy.events import Event, UntilTEvent, UntilNEvent, InflationEvent
from primpy.events import AfterInflationEndEvent, Phi0Event
from primpy.time.inflation import InflationEquationsT
from primpy.efolds.inflation import InflationEquationsN
from primpy.initialconditions import InflationStartIC
from primpy.solver import solve


def test_not_implemented_errors():
    ev = Event(None)
    with pytest.raises(NotImplementedError, match="Event class must define __call__."):
        ev(x=0, y=np.zeros(3))


@pytest.mark.parametrize('K', [-1, 0, +1])
def test_UntilTEvent(K):
    pot = QuadraticPotential(Lambda=np.sqrt(6e-6))
    t_i = 7e4
    N_i = 10
    phi_i = 17
    t_end = 1e6
    for eq in [InflationEquationsT(K=K, potential=pot),
               InflationEquationsN(K=K, potential=pot, track_time=True)]:
        ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
        ev = [UntilTEvent(eq, t_end)]
        sol = solve(ic=ic, events=ev)
        assert sol.t[-1] == approx(t_end)
        assert sol.t_events['UntilT'][-1] == approx(t_end)


@pytest.mark.parametrize('K', [-1, 0, +1])
@pytest.mark.parametrize('Eq', [InflationEquationsT, InflationEquationsN])
def test_UntilNEvent(K, Eq):
    pot = QuadraticPotential(Lambda=np.sqrt(6e-6))
    t_i = 7e4
    N_i = 10
    phi_i = 17
    N_end = 73
    eq = Eq(K=K, potential=pot)
    ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
    ev = [UntilNEvent(eq, N_end)]
    sol = solve(ic=ic, events=ev)
    assert sol.N[-1] == approx(N_end)
    assert sol.N_events['UntilN'][-1] == approx(N_end)


@pytest.mark.parametrize('K', [-1, 0, +1])
@pytest.mark.parametrize('Lambda', [1, np.sqrt(6e-6)])
@pytest.mark.parametrize('Eq', [InflationEquationsT, InflationEquationsN])
def test_InflationEvent(K, Lambda, Eq):
    t_i = 7e4
    N_i = 10
    phi_i = 17
    pot = QuadraticPotential(Lambda=Lambda)
    eq = Eq(K=K, potential=pot)
    ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
    ev = [InflationEvent(eq, +1, terminal=False),
          InflationEvent(eq, -1, terminal=True)]
    sol = solve(ic=ic, events=ev)
    assert np.isfinite(sol.N_beg)
    assert np.isfinite(sol.N_end)
    assert sol.w[0] == approx(-1 / 3)
    assert sol.w[-1] == approx(-1 / 3)
    assert np.all(sol.w[1:-1] < -1 / 3)


@pytest.mark.parametrize('K', [-1, 0, +1])
@pytest.mark.parametrize('Eq', [InflationEquationsT, InflationEquationsN])
def test_AfterInflationEndEvent(K, Eq):
    pot = QuadraticPotential(Lambda=np.sqrt(6e-6))
    t_i = 7e4
    N_i = 10
    phi_i = 17
    eq = Eq(K=K, potential=pot)
    ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
    ev = [InflationEvent(eq, +1, terminal=False),
          InflationEvent(eq, -1, terminal=False),
          AfterInflationEndEvent(eq)]
    sol = solve(ic=ic, events=ev)
    assert np.isfinite(sol.N_beg)
    assert np.isfinite(sol.N_end)
    assert sol.w[-1] == approx(0)
    assert np.all(sol.w[1:-1] < 0)
    assert sol.N_events['Inflation_dir-1_term0'].size == 1
    assert (sol.N_events['Inflation_dir-1_term0'][0] <
            sol.N_events['AfterInflationEnd_dir1_term1'][0])


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt:RuntimeWarning")
@pytest.mark.parametrize('K', [-1, 0, +1])
@pytest.mark.parametrize('Eq', [InflationEquationsT, InflationEquationsN])
def test_Phi0Event(K, Eq):
    pot = QuadraticPotential(Lambda=np.sqrt(6e-6))
    t_i = 7e4
    N_i = 12
    phi_i = 17
    eq = Eq(K=K, potential=pot)
    ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
    ev = [InflationEvent(eq, +1, terminal=False),
          InflationEvent(eq, -1, terminal=False),
          Phi0Event(eq)]
    sol = solve(ic=ic, events=ev)
    assert np.isfinite(sol.N_beg)
    assert np.isfinite(sol.N_end)
    assert sol.N_events['Inflation_dir-1_term0'].size == 1
    assert (sol.N_events['Inflation_dir-1_term0'][0] <
            sol.N_events['Phi0_dir0_term1'][0])
    assert sol.phi[-1] == approx(0)
