#!/usr/bin/env python
"""Tests for `primpy.potential` module."""
import pytest
import numpy as np
from primpy.potentials import QuadraticPotential, StarobinskyPotential
from primpy.events import InflationEvent, UntilNEvent
from primpy.time.initialconditions import InflationStartIC_NiPi
from primpy.solver import solve


@pytest.mark.filterwarnings("ignore:.*Inflation start not determined.*:UserWarning")
@pytest.mark.filterwarnings("ignore:.*Inflation end not determined.*:UserWarning")
def test_equations_sol_ordering_after_postprocessing():
    t_i = 1e4
    N_i = 10
    phis = [17, 6]
    pots = [QuadraticPotential(m=6e-6), StarobinskyPotential(Lambda=5e-2)]
    for K in [-1, 0, +1]:
        for i, pot in enumerate(pots):
            phi_i = phis[i]

            # integration forwards in time:
            ic_foreward = InflationStartIC_NiPi(t_i=t_i, N_i=N_i, phi_i=phi_i, K=K, potential=pot)
            # integration backward in time:
            ic_backward = InflationStartIC_NiPi(t_i=t_i, N_i=N_i, phi_i=phi_i, K=K, potential=pot,
                                                t_end=1)

            # stop at end of inflation:
            ev_foreward = [InflationEvent(ic_foreward.equations, +1, terminal=False),
                           InflationEvent(ic_foreward.equations, -1, terminal=True)]
            # stop when N = 0:
            ev_backward = [UntilNEvent(ic_backward.equations, value=0, terminal=True)]

            bist_foreward = solve(ic=ic_foreward, events=ev_foreward)
            bist_backward = solve(ic=ic_backward, events=ev_backward)

            # time grows monotonically forwards in time
            assert np.all(np.diff(bist_foreward.x) > 0)
            assert np.all(np.diff(bist_foreward.t) > 0)
            # e-folds grow monotonically forwards in time
            assert np.all(np.diff(bist_foreward.y[0]) > 0)
            assert np.all(np.diff(bist_foreward.N) > 0)
            # phi shrinks monotonically forwards in time (from start to end of inflation)
            assert np.all(np.diff(bist_foreward.y[1]) < 0)
            assert np.all(np.diff(bist_foreward.phi) < 0)

            # time shrinks monotonically backwards in time
            assert np.all(np.diff(bist_backward.x) < 0)
            assert np.all(np.diff(bist_backward.t) < 0)
            # e-folds shrink monotonically backwards in time
            assert np.all(np.diff(bist_backward.y[0]) < 0)
            assert np.all(np.diff(bist_backward.N) < 0)
            # phi grows monotonically backwards in time (before start of inflation)
            assert np.all(np.diff(bist_backward.y[1]) > 0)
            assert np.all(np.diff(bist_backward.phi) > 0)
