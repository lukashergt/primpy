#!/usr/bin/env python
""":mod:`primpy.solver`: general setup for running `solve_ivp`."""
import numpy as np
from scipy import integrate, interpolate
import pyoscode
from primpy.time.perturbations import CurvaturePerturbationT
from primpy.efolds.perturbations import CurvaturePerturbationN


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


def solve_oscode(background, k, **kwargs):
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
    rtol = kwargs.pop('rtol', 5e-5)
    fac = kwargs.pop('fac', 100)
    x0_1, dx0_1, x0_2, dx0_2 = kwargs.pop('ic', [1, 0, 0, 1])
    if isinstance(k, int) or isinstance(k, float):
        k = np.atleast_1d(k)
        return_pps = False
    else:
        return_pps = True
    pps = np.zeros_like(k, dtype=float)
    # stop integration sufficiently after mode has crossed the horizon (lazy for loop):
    j = 0
    for i, ki in enumerate(k):
        for j in range(j, background.x.size):
            if background.aH[j] / ki > fac:
                if background.independent_variable == 't':
                    pert = CurvaturePerturbationT(background=background, k=ki)
                elif background.independent_variable == 'N':
                    pert = CurvaturePerturbationN(background=background, k=ki)
                else:
                    raise NotImplementedError()
                logf = np.log(pert.ms_frequency)
                damp = pert.ms_damping
                sol1 = pyoscode.solve(ts=background.x, ti=background.x[0], tf=background.x[j],
                                      ws=logf, logw=True,
                                      gs=damp, logg=False,
                                      x0=x0_1, dx0=dx0_1, rtol=rtol)
                sol2 = pyoscode.solve(ts=background.x, ti=background.x[0], tf=background.x[j],
                                      ws=logf, logw=True,
                                      gs=damp, logg=False,
                                      x0=x0_2, dx0=dx0_2, rtol=rtol)
                pert = pert.sol(sol=pert, sol1=sol1, sol2=sol2)
                pps[i] = pert.PPS_RST
                break
    if return_pps:
        return pps
    else:
        return pert


def solve_oscode_N(background, k, **kwargs):
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
    rtol = kwargs.pop('rtol', 1e-4)
    x0_1, dx0_1, x0_2, dx0_2 = kwargs.pop('ic', [1, 0, 0, 1])
    if isinstance(k, int) or isinstance(k, float):
        k = np.atleast_1d(k)
        return_pps = False
    else:
        return_pps = True
    pps = np.zeros_like(k)
    # stop integration sufficiently after mode has crossed the horizon (lazy for loop):
    j = 0
    for i, ki in enumerate(k):
        for j in range(j, background.N.size):
            if background.aH[j] / ki > 1000:
                # idx_stop[i] = j
                pert = CurvaturePerturbationN(background=background, k=ki)
                logf = np.log(pert.ms_frequency)
                damp = pert.ms_damping
                N2logf = interpolate.interp1d(background.N, logf, kind='cubic')
                N2damp = interpolate.interp1d(background.N, damp, kind='cubic')
                N = np.linspace(background.N[0], background.N[j], 100000)
                logf = N2logf(N)
                damp = N2damp(N)
                sol1 = pyoscode.solve(ts=N, ti=N[0], tf=N[-1],
                                      ws=logf, logw=True,
                                      gs=damp, logg=False,
                                      x0=x0_1, dx0=dx0_1, rtol=rtol)
                sol2 = pyoscode.solve(ts=N, ti=N[0], tf=N[-1],
                                      ws=logf, logw=True,
                                      gs=damp, logg=False,
                                      x0=x0_2, dx0=dx0_2, rtol=rtol)
                pert = pert.sol(sol=pert, sol1=sol1, sol2=sol2)
                pps[i] = pert.PPS_RST
                break
    if return_pps:
        return pps
    else:
        return pert


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
