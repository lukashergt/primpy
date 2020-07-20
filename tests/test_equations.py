#!/usr/bin/env python
"""Tests for `primpy.equation` module."""
import pytest
import numpy as np
from numpy.testing import assert_equal
from primpy.potentials import QuadraticPotential, StarobinskyPotential
from primpy.events import InflationEvent, UntilNEvent, CollapseEvent
from primpy.time.inflation import InflationEquationsT
from primpy.efolds.inflation import InflationEquationsN
from primpy.equations import Equations
from primpy.initialconditions import InflationStartIC
from primpy.solver import solve


def test_not_implemented_errors():
    eq = Equations()
    with pytest.raises(NotImplementedError, match="Equations class must define __call__."):
        eq(x=0, y=np.zeros(3))


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt:RuntimeWarning")
@pytest.mark.parametrize('K', [-1, 0, +1])
@pytest.mark.parametrize('phi_i, pot', [(17, QuadraticPotential(mass=6e-6)),
                                        (6, StarobinskyPotential(Lambda=5e-2))])
def test_equations_sol_ordering_after_postprocessing(K, phi_i, pot):
    t_i = 7e4
    N_i = 12
    for eq in [InflationEquationsT(K=K, potential=pot),
               InflationEquationsN(K=K, potential=pot, track_time=True)]:
        # integration forwards in time:
        ic_forwards = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
        # integration backward in time:
        ic_backward = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i, x_end=1)

        # stop at end of inflation:
        ev_forwards = [InflationEvent(ic_forwards.equations, +1, terminal=False),
                       InflationEvent(ic_forwards.equations, -1, terminal=True)]
        # stop when N = 0:
        ev_backward = [UntilNEvent(ic_backward.equations, value=1, terminal=True)]

        b_forwards = solve(ic=ic_forwards, events=ev_forwards)
        b_backward = solve(ic=ic_backward, events=ev_backward)

        # time and e-folds grow monotonically forwards in time
        assert np.all(np.diff(b_forwards.x) > 0)
        assert np.all(np.diff(b_forwards.t) > 0)
        assert np.all(np.diff(b_forwards.N) > 0)
        # phi shrinks monotonically forwards in time (from start to end of inflation)
        assert np.all(np.diff(b_forwards.y[eq.idx['phi']]) < 0)
        assert np.all(np.diff(b_forwards.phi) < 0)

        # time and e-folds shrink monotonically backwards in time
        assert np.all(np.diff(b_backward.x) < 0)
        assert np.all(np.diff(b_backward.t) < 0)
        assert np.all(np.diff(b_backward.N) < 0)
        # phi grows monotonically backwards in time (before start of inflation)
        assert np.all(np.diff(b_backward.y[eq.idx['phi']]) > 0)
        assert np.all(np.diff(b_backward.phi) > 0)


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt:RuntimeWarning")
@pytest.mark.parametrize('K', [-1, 0, +1])
def test_equations_sol_events(K):
    t_i = 7e4
    N_i = 10
    phi_i = 17
    pot = QuadraticPotential(Lambda=np.sqrt(6e-6))
    N_end = 80
    for eq in [InflationEquationsT(K=K, potential=pot),
               InflationEquationsN(K=K, potential=pot, track_time=True)]:
        ic = InflationStartIC(equations=eq, N_i=N_i, phi_i=phi_i, t_i=t_i)
        ev = [CollapseEvent(ic.equations),
              InflationEvent(ic.equations, +1, terminal=False),
              InflationEvent(ic.equations, -1, terminal=False),
              UntilNEvent(ic.equations, N_end)]
        sol = solve(ic=ic, events=ev)

        assert hasattr(sol, 't_events')
        assert hasattr(sol, 'N_events')
        assert hasattr(sol, 'phi_events')

        for key, value in sol.y_events.items():
            if value.size == 0:
                assert_equal(sol.t_events[key], value)
                assert_equal(sol.N_events[key], value)
                assert_equal(sol.phi_events[key], value)
                if isinstance(eq, InflationEquationsT):
                    assert hasattr(sol, 'dphidt_events')
                    assert_equal(sol.dphidt_events[key], value)
                elif isinstance(eq, InflationEquationsN):
                    assert hasattr(sol, 'dphidN_events')
                    assert_equal(sol.dphidN_events[key], value)
            else:
                assert_equal(sol.phi_events[key], value[:, eq.idx['phi']])
                if isinstance(eq, InflationEquationsT):
                    assert_equal(sol.N_events[key], value[:, eq.idx['N']])
                    assert hasattr(sol, 'dphidt_events')
                    assert_equal(sol.dphidt_events[key], value[:, eq.idx['dphidt']])
                elif isinstance(eq, InflationEquationsN):
                    assert_equal(sol.t_events[key], value[:, eq.idx['t']])
                    assert hasattr(sol, 'dphidN_events')
                    assert_equal(sol.dphidN_events[key], value[:, eq.idx['dphidN']])
