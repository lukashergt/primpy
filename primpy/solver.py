#!/usr/bin/env python
""":mod:`primpy.solver`: general setup for running :func:`scipy.integrate.solve_ivp`."""
import numpy as np
from scipy import integrate


def solve(ic, *args, **kwargs):
    """Run :func:`scipy.integrate.solve_ivp` and store information in `sol` for post-processing.

    This is a wrapper around :func:`scipy.integrate.solve_ivp`, with easier
    reusable objects for the equations and initial conditions.

    Parameters
    ----------
        ic : primordial.initialconditions.InitialConditions
            Initial conditions specifying relevant equations, variables, and
            initial numerical values.

    All other arguments are identical to :func:`scipy.integrate.solve_ivp`.

    Returns
    -------
        sol : Bunch object
            Solution to the inverse value problem.
            Monkey-patched version of the Bunch type usually returned by
            :func:`scipy.integrate.solve_ivp`.

    `(c)` modified from "primordial" by Will Handley.

    """
    events = kwargs.pop('events', [])
    y0 = np.zeros(len(ic.equations.idx))
    method = kwargs.pop('method', 'RK45')
    rtol = kwargs.pop('rtol', 1e-10)
    atol = kwargs.pop('atol', 1e-12)
    ic(y0=y0, rtol=rtol, atol=atol, **kwargs)
    sol = integrate.solve_ivp(ic.equations, (ic.x_ini, ic.x_end), y0, events=events,
                              method=method, rtol=rtol, atol=atol, *args, **kwargs)
    sol.event_keys = [e.name for e in events]
    return ic.equations.sol(sol)
