#!/usr/bin/env python
""":mod:`primpy.solver`: general setup for running `solve_ivp`."""
import numpy as np
from scipy import integrate


def solve(ic, *args, **kwargs):
    """Run `solve_ivp` and store information in `sol` for post-processing.

    This is a wrapper around ``scipy.integrate.solve_ivp``, with easier
    reusable objects for the equations and initial conditions.

    Parameters
    ----------
    ic : primordial.initialconditions.InitialConditions
        Initial conditions specifying relevant equations, variables, and
        initial numerical values.

    All other arguments are identical to ``scipy.integrate.solve_ivp``

    Returns
    -------
    sol : Bunch object same as returned by `scipy.integrate.solve_ivp`
        Monkey-patched version of the Bunch type usually returned by `solve_ivp`.

    (c) modified from "primordial" by Will Handley.
    """
    events = kwargs.pop('events', [])
    y0 = np.zeros(len(ic.equations.idx))
    ic(y0=y0, **kwargs)
    sol = integrate.solve_ivp(ic.equations, (ic.x_ini, ic.x_end), y0, events=events,
                              *args, **kwargs)
    sol.event_keys = [e.name for e in events]
    return ic.equations.sol(sol)
