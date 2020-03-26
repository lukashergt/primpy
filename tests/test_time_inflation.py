#!/usr/bin/env python
"""Tests for `primpy.potential` module."""
import pytest
import numpy as np
from primpy.potentials import QuadraticPotential
from primpy.events import InflationEvent, UntilNEvent
from primpy.time.initialconditions import InflationStartIC_NiPi
from primpy.solver import solve


def nan_inflation_end(background_solution):
    assert not np.isfinite(background_solution.t_end)
    assert not np.isfinite(background_solution.N_end)
    assert not np.isfinite(background_solution.phi_end)
    assert not np.isfinite(background_solution.V_end)
    assert not np.isfinite(background_solution.N_tot)
    assert not hasattr(background_solution, 'inflation_mask')


def test_postprocessing_inflation_end_warnings():
    t_i = 1e4
    N_i = 10
    phi_i = 17
    pot = QuadraticPotential(m=6e-6)
    for K in [-1, 0, +1]:
        # set t_end earlier to trigger "inflation not ended warning:
        ic_early_end = InflationStartIC_NiPi(t_i=t_i, N_i=N_i, phi_i=phi_i, K=K, potential=pot,
                                             t_end=1e6)
        ev = [InflationEvent(ic_early_end.equations, +1, terminal=False),
              InflationEvent(ic_early_end.equations, -1, terminal=True)]
        with pytest.warns(UserWarning, match="Inflation has not ended."):
            bist = solve(ic=ic_early_end, events=ev)
        nan_inflation_end(background_solution=bist)

        # no passing of InflationEvent(-1), i.e. inflation end not recorded
        ic = InflationStartIC_NiPi(t_i=t_i, N_i=N_i, phi_i=phi_i, K=K, potential=pot, t_end=1e7)
        ev_no_end = [InflationEvent(ic.equations, +1, terminal=False)]
        with pytest.warns(UserWarning, match="Inflation end not determined. In order to"):
            bist = solve(ic=ic, events=ev_no_end)
        nan_inflation_end(background_solution=bist)

        # no inflation end recorded, despite not being in inflation, despite using event tracking
        # e.g. when integrating backwards in time
        ic_backward = InflationStartIC_NiPi(t_i=t_i, N_i=N_i, phi_i=phi_i, K=K, potential=pot,
                                            t_end=1)
        ev_backward = [UntilNEvent(ic_backward.equations, value=0),
                       InflationEvent(ic_backward.equations, -1, terminal=True)]
        with pytest.warns(UserWarning, match="Inflation end not determined."):
            bist_backward = solve(ic=ic_backward, events=ev_backward)
        nan_inflation_end(background_solution=bist_backward)
