#!/usr/bin/env python
"""Tests for `primpy.equation` module."""
import pytest
import numpy as np
from numpy.testing import assert_equal
from primpy.potentials import QuadraticPotential, StarobinskyPotential
from primpy.events import InflationEvent, UntilNEvent, CollapseEvent
from primpy.time.initialconditions import InflationStartIC_NiPi
from primpy.solver import solve


@pytest.mark.filterwarnings("ignore:.*Inflation start not determined.*:UserWarning")
@pytest.mark.filterwarnings("ignore:.*Inflation end not determined.*:UserWarning")
def test_equations_sol_ordering_after_postprocessing():
    t_i = 1e4
    N_i = 10
    phis = [17, 6]
    pots = [QuadraticPotential(mass=6e-6), StarobinskyPotential(Lambda=5e-2)]
    for K in [-1, 0, +1]:
        for i, pot in enumerate(pots):
            phi_i = phis[i]

            # integration forwards in time:
            ic_forwards = InflationStartIC_NiPi(t_i=t_i, N_i=N_i, phi_i=phi_i, K=K, potential=pot)
            # integration backward in time:
            ic_backward = InflationStartIC_NiPi(t_i=t_i, N_i=N_i, phi_i=phi_i, K=K, potential=pot,
                                                t_end=1)

            # stop at end of inflation:
            ev_forwards = [InflationEvent(ic_forwards.equations, +1, terminal=False),
                           InflationEvent(ic_forwards.equations, -1, terminal=True)]
            # stop when N = 0:
            ev_backward = [UntilNEvent(ic_backward.equations, value=0, terminal=True)]

            bist_forwards = solve(ic=ic_forwards, events=ev_forwards)
            bist_backward = solve(ic=ic_backward, events=ev_backward)

            # time grows monotonically forwards in time
            assert np.all(np.diff(bist_forwards.x) > 0)
            assert np.all(np.diff(bist_forwards.t) > 0)
            # e-folds grow monotonically forwards in time
            assert np.all(np.diff(bist_forwards.y[0]) > 0)
            assert np.all(np.diff(bist_forwards.N) > 0)
            # phi shrinks monotonically forwards in time (from start to end of inflation)
            assert np.all(np.diff(bist_forwards.y[1]) < 0)
            assert np.all(np.diff(bist_forwards.phi) < 0)

            # time shrinks monotonically backwards in time
            assert np.all(np.diff(bist_backward.x) < 0)
            assert np.all(np.diff(bist_backward.t) < 0)
            # e-folds shrink monotonically backwards in time
            assert np.all(np.diff(bist_backward.y[0]) < 0)
            assert np.all(np.diff(bist_backward.N) < 0)
            # phi grows monotonically backwards in time (before start of inflation)
            assert np.all(np.diff(bist_backward.y[1]) > 0)
            assert np.all(np.diff(bist_backward.phi) > 0)


def test_equations_sol_events():
    t_i = 1e4
    N_i = 10
    phi_i = 17
    pot = QuadraticPotential(mass=6e-6)
    N_end = 80
    for K in [-1, 0, +1]:
        ic = InflationStartIC_NiPi(t_i=t_i, N_i=N_i, phi_i=phi_i, K=K, potential=pot)
        ev = [CollapseEvent(ic.equations),
              InflationEvent(ic.equations, +1, terminal=False),
              InflationEvent(ic.equations, -1, terminal=False),
              UntilNEvent(ic.equations, N_end)]
        bist = solve(ic=ic, events=ev)

        assert hasattr(bist, 't_events')
        assert hasattr(bist, 'N_events')
        assert hasattr(bist, 'phi_events')
        assert hasattr(bist, 'dphidt_events')
        assert hasattr(bist, 'eta_events')

        for key, value in bist.y_events.items():
            if value.size == 0:
                assert_equal(bist.t_events[key], value)
                assert_equal(bist.N_events[key], value)
                assert_equal(bist.phi_events[key], value)
                assert_equal(bist.dphidt_events[key], value)
                assert_equal(bist.eta_events[key], value)
            else:
                assert_equal(bist.N_events[key], value[:, 0])
                assert_equal(bist.phi_events[key], value[:, 1])
                assert_equal(bist.dphidt_events[key], value[:, 2])
                assert_equal(bist.eta_events[key], value[:, 3])
