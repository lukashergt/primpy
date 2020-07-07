#!/usr/bin/env python
""":mod:`primpy.solver`: general setup for running `solve_ivp`."""
import numpy as np
from scipy import integrate
import pyoscode
from primpy.time.perturbations import CurvaturePerturbationT


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
    rtol = kwargs.pop('rtol', 1e-6)
    atol = kwargs.pop('atol', 1e-10)
    ic(y0=y0, rtol=rtol, atol=atol, **kwargs)
    sol = integrate.solve_ivp(ic.equations, (ic.x_ini, ic.x_end), y0, events=events,
                              rtol=rtol, atol=atol, *args, **kwargs)
    sol.event_keys = [e.name for e in events]
    return ic.equations.sol(sol)


def solve_oscode(background, k):
    """Run `pyoscode.solve` and store information for post-processing.

    This is a wrapper around ``pyoscode.solve`` to calculate the solution to
    the Mukhanov-Sasaki equation.

    Parameters
    ----------
    background : Bunch object as returned by `primpy.solver.solve`
        Solution to the inflationary background equations used to calculate
        the frequency and damping term passed to oscode.
    k : int, float
        Comoving wavenumber used to evolve the Mukhanov-Sasaki equation.

    Returns
    -------
    sol : Bunch object similar to that returned by `scipy.integrate.solve_ivp`
        Monkey-patched version of the Bunch type returned by `solve_ivp`,
        containing the primordial power spectrum value corresponding to the
        wavenumber `k`.
    """
    pert = CurvaturePerturbationT(background=background, k=k)
    sol1 = pyoscode.solve(ts=background.t, ws=pert.ms_frequency, gs=pert.ms_damping,
                          ti=background.t[0], tf=background.t[-1], x0=1, dx0=0,
                          logw=False, logg=False, rtol=1e-6)
    sol2 = pyoscode.solve(ts=background.t, ws=pert.ms_frequency, gs=pert.ms_damping,
                          ti=background.t[0], tf=background.t[-1], x0=0, dx0=1,
                          logw=False, logg=False, rtol=1e-6)
    return pert.sol(sol=pert, sol1=sol1, sol2=sol2)


def solve_pps(background, ks):
    """Run `pyoscode.solve` for a range of wavenumbers to get the PPS.

    This is a wrapper around ``pyoscode.solve`` to calculate the primordial
    power spectrum (PPS) for a range of wavenumbers `ks`.

    Parameters
    ----------
    background : Bunch object as returned by `primpy.solver.solve`
        Solution to the inflationary background equations used to calculate
        the frequency and damping term passed to oscode.
    ks : np.ndarray
        Array of comoving wavenumbers.

    Returns
    -------
    PPS : np.ndarray
        Array of the primordial power spectrum matching to the
        wavenumbers `ks`.
    """
    return np.array([solve_oscode(background, k).PPS_RST for k in ks])
