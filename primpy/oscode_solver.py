#!/usr/bin/env python
""":mod:`primpy.oscode_solver`: setup for running :func:`pyoscode.solve`."""
import numpy as np
import pyoscode
from primpy.perturbations import PrimordialPowerSpectrum
from primpy.time.perturbations import PerturbationT
from primpy.efolds.perturbations import PerturbationN


def solve_oscode(background, k, **kwargs):
    """Run :func:`pyoscode.solve` and store information for post-processing.

    This is a wrapper around :func:`pyoscode.solve` to calculate the solution to
    the Mukhanov-Sasaki equation.

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
        fac : int, float
            Integration of the mode evolution stops when the considered scale k
            exceeds the comoving Hubble horizon by a factor of `fac`, i.e. when
            `aH / k > fac`.
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
    fac = kwargs.pop('fac', 100)
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
    j = 2
    for i, ki in enumerate(k):
        for j in range(j, b.x.size):
            if b.logaH[j] - np.log(ki) > np.log(fac):
                if b.independent_variable == 't':
                    p = PerturbationT(background=b, k=ki, idx_end=j+1, **kwargs)
                elif b.independent_variable == 'N':
                    p = PerturbationN(background=b, k=ki, idx_end=j+1, **kwargs)
                else:
                    raise NotImplementedError()
                oscode_sol = []
                for mode in [p.scalar, p.tensor]:
                    for num in range(2):
                        oscode_sol.append(pyoscode.solve(ts=b.x[:j+1], ti=b.x[0], tf=b.x[j],
                                                         ws=np.log(mode.ms_frequency), logw=True,
                                                         gs=mode.ms_damping, logg=False,
                                                         x0=y0[2 * num] * ki,
                                                         dx0=y0[2 * num + 1] * ki**2,
                                                         rtol=rtol, even_grid=even_grid))
                p.oscode_postprocessing(oscode_sol=oscode_sol)
                if ki < 1 and b.K == +1 and drop_closed_large_scales:
                    p.scalar.P_s_RST = 1e-30
                for vac in vacuum:
                    getattr(PPS, 'P_s_%s' % vac)[i] = getattr(p.scalar, 'P_s_%s' % vac)
                    getattr(PPS, 'P_t_%s' % vac)[i] = getattr(p.tensor, 'P_t_%s' % vac)
                break
    if return_pps:
        return PPS
    else:
        return p
