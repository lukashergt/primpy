#!/usr/bin/env python
""":mod:`primpy.oscode_solver`: setup for running :func:`pyoscode.solve`."""
import numpy as np
import pyoscode
from primpy.perturbations import PrimordialPowerSpectrum
from primpy.time.perturbations import PerturbationT
from primpy.efolds.perturbations import PerturbationN


def solve_oscode(background, k, **kwargs):
    """Run :func:`pyoscode.solve` and store information for post-processing.

    This is a wrapper around :func:`pyoscode.solve` to calculate the solution
    to the Mukhanov-Sasaki equation.

    Parameters
    ----------
        background : Bunch object as returned by :func:`primpy.solver.solve`
            Solution to the inflationary background equations used to calculate
            the frequency and damping term passed to oscode.
        k : int, float, np.ndarray
            Comoving wavenumber used to evolve the Mukhanov-Sasaki equation.

    Keyword args
    ------------
        y0 : (float, float, float, float)
            Initial values (y0_1, dy0_1, y0_2, dy0_2) of perturbations and
            their derivatives for two independent solutions. The perturbations
            (y0_1, y0_2) are scaled with `k` and their derivatives with `k**2`
            in order to produce freeze-out values of about order(~1).
            default : determined by input inflationary potential
        rtol : float
            Tolerance passed to pyoscode.
            default : 5e-5
        fac_beg : int, float
            Integration of the mode evolution starts when the considered
            scale 1/k is within a factor of `fac_beg` of the comoving Hubble
            horizon, i.e. when `1/k > 1/aH / fac_beg`.
            `fac_beg=0` starts integration immediately.
            default : 0
        fac_end : int, float
            Integration of the mode evolution stops when the considered
            scale 1/k exceeds the comoving Hubble horizon by a factor of
            `fac_end`, i.e. when `1/k > 1/aH * fac_end`.
            default : 100
        even_grid : bool
            Set this to True if the grid of the independent variable is
            equally spaced.
            default : False
        vacuum : tuple
            Set of vacuum initial conditions to be computed.
            Choose any of ('RST', ).
            default : ('RST', )
        drop_closed_large_scales : bool
            If true, this will set the PPS for closed universes on comoving
            scales of `k < 1` to close to zero (1e-30). Strictly speaking, the
            PPS for closed universes is only defined for rational numbers
            `k > 2`.
            default : True

    Returns
    -------
        sol : Bunch object
            Solution to the inverse value problem, containing the primordial
            power spectrum value corresponding to the wavenumber `k`.
            Monkey-patched version of the Bunch type usually returned by
            :func:`scipy.integrate.solve_ivp`.

    """
    assert 'tol' not in kwargs
    y0 = kwargs.pop('y0', background.potential.perturbation_ic)
    rtol = kwargs.pop('rtol', 5e-5)
    fac_beg = kwargs.pop('fac_beg', 0)
    fac_end = kwargs.pop('fac_end', 100)
    even_grid = kwargs.pop('even_grid', False)
    vacuum = kwargs.get('vacuum', ('RST',))
    drop_closed_large_scales = kwargs.pop('drop_closed_large_scales', True)
    b = background
    if isinstance(k, int) or isinstance(k, float):
        k = np.atleast_1d(k)
        return_pps = False
    else:
        return_pps = True
    PPS = PrimordialPowerSpectrum(background=b, k=k, **kwargs)
    # stop integration sufficiently after mode has crossed the horizon (lazy for loop):
    for i, ki in enumerate(k):
        if fac_beg == 0:
            idx_beg = 0
        else:
            idx_beg = np.argwhere(np.log(ki) - b.logaH > np.log(fac_beg)).ravel()
            # set idx_beg to 0 if horizon starts out too small:
            idx_beg = 0 if idx_beg.size == 0 else idx_beg[-1]
        idx_end = np.argwhere(b.logaH - np.log(ki) > np.log(fac_end)).ravel()[0]
        # set minimum for idx_end, needed e.g. in KD for superhorizon modes:
        idx_end = idx_end if idx_end - idx_beg > b.logaH.size//20 else idx_beg + b.logaH.size//20
        if b.independent_variable == 't':
            p = PerturbationT(background=b, k=ki, idx_beg=idx_beg, idx_end=idx_end, **kwargs)
        elif b.independent_variable == 'N':
            p = PerturbationN(background=b, k=ki, idx_beg=idx_beg, idx_end=idx_end, **kwargs)
        else:
            raise NotImplementedError()
        oscode_sol = []
        for mode in [p.scalar, p.tensor]:
            for num in range(2):
                oscode_sol.append(pyoscode.solve(ts=b.x[idx_beg:idx_end+1],
                                                 ti=b.x[idx_beg], tf=b.x[idx_end],
                                                 ws=mode.ms_frequency, logw=False,
                                                 gs=mode.ms_damping, logg=False,
                                                 x0=y0[2*num]*ki,
                                                 dx0=y0[2*num+1]*ki**2,
                                                 rtol=rtol, even_grid=even_grid))
        p.oscode_postprocessing(oscode_sol=oscode_sol)
        if ki < 1 and b.K == +1 and drop_closed_large_scales:
            p.scalar.P_s_RST = 1e-30
        for vac in vacuum:
            getattr(PPS, 'P_s_%s' % vac)[i] = getattr(p.scalar, 'P_s_%s' % vac)
            getattr(PPS, 'P_t_%s' % vac)[i] = getattr(p.tensor, 'P_t_%s' % vac)
    if return_pps:
        return PPS
    else:
        return p
