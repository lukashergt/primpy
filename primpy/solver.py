import numpy as np
from scipy import integrate


def solve(ic, *args, **kwargs):
    """Wrapper for solving differential equations in the primordial Universe.

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
    y0 = np.zeros(len(ic.equations.idx))
    method = kwargs.pop('method', 'DOP853')
    ic(y0=y0)
    events = kwargs.pop('events', [])
    sol = integrate.solve_ivp(ic.equations, (ic.x_ini, ic.x_end), y0, method=method, events=events,
                              *args, **kwargs)
    sol.event_keys = [e.name for e in events]
    return ic.equations.sol(sol)
