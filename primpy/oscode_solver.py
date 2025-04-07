"""Setup for running :func:`pyoscode.solve`."""
import numpy as np
import pyoscode
from primpy.perturbations import PrimordialPowerSpectrum
from primpy.time.perturbations import PerturbationT
from primpy.efolds.perturbations import PerturbationN


def solve_oscode(background, k,
                 vacuum=None, drop_closed_large_scales=True,
                 fac_beg=0, fac_end=100,
                 rtol=5e-5, even_grid=False, num_eval=0,
                 **kwargs):
    """Run :func:`pyoscode.solve` and store information for post-processing.

    This is a wrapper around :func:`pyoscode.solve` to calculate the solution
    to the Mukhanov-Sasaki equation.

    Parameters
    ----------
    background : Bunch object same as returned by :func:`scipy.integrate.solve_ivp`
        Bunch object as returned by :func:`primpy.solver.solve`.
        Solution to the inflationary background equations used to calculate
        the frequency and damping term passed to oscode.
    k : int, float, np.ndarray
        Comoving wavenumber used to evolve the Mukhanov-Sasaki equation.
    vacuum : tuple
        Set of vacuum initial conditions to be computed.
        Choose any of ('k', 'HD', 'RST').
        default : ('RST', )
    drop_closed_large_scales : bool
        If true, this will set the PPS for closed universes on comoving
        scales of `k < 1` to close to zero (1e-30). Strictly speaking, the
        PPS for closed universes is only defined for rational numbers
        `k > 2`.
        default : True
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
    rtol : float
        Tolerance passed to pyoscode.
        default : 5e-5
    even_grid : bool
        Set this to True if the grid of the independent variable is
        equally spaced.
        default : False
    num_eval : int
        Number of interpolation points used for dense output.
        This number is applied dynamically to the range determined from the
        parameters `fac_beg` and `fac_end`.
        Zero means no dense output.
        default : 0

    Other Parameters
    ----------------
    y0 : (float, float, float, float)
        Initial values (y0_1, dy0_1, y0_2, dy0_2) of perturbations and
        their derivatives for two independent solutions. The perturbations
        (y0_1, y0_2) are scaled with `k` and their derivatives with `k**2`
        in order to produce freeze-out values of about order(~1).
        default : determined by input inflationary potential

    Returns
    -------
    sol : Bunch object same as returned by :func:`scipy.integrate.solve_ivp`
        Solution to the inverse value problem, containing the primordial
        power spectrum value corresponding to the wavenumber `k`.
        Monkey-patched version of the Bunch type usually returned by
        :func:`scipy.integrate.solve_ivp`.

    """
    assert 'tol' not in kwargs
    b = background
    vacuum = ('RST', ) if vacuum is None else vacuum
    y0 = kwargs.pop('y0', b.potential.perturbation_ic)
    if isinstance(k, int) or isinstance(k, float):
        k = np.atleast_1d(k)
        return_pps = False
    else:
        return_pps = True
    PPS = PrimordialPowerSpectrum(background=b, k=k, vacuum=vacuum)
    # stop integration sufficiently after mode has crossed the horizon (lazy for loop):
    for i, ki in enumerate(k):
        if fac_beg == 0:
            idx_beg = 0
        else:
            idx_beg = np.argwhere(np.log(ki) - b._logaH > np.log(fac_beg)).ravel()
            # set idx_beg to 0 if horizon starts out too small:
            idx_beg = 0 if idx_beg.size == 0 else idx_beg[-1]
        idx_end = np.argwhere(b._logaH - np.log(ki) > np.log(fac_end)).ravel()[0]
        if b._N[idx_end] - b._N[idx_beg] < 10:
            # For KD, for very large modes it can happen that they never go sub-horizon, and
            # therefore `idx_end` will be set equal (or close to equal) to `idx_beg`. These modes
            # nonetheless need to be evolved a bit past inflationstart before they properly freeze
            # in. Hence, we additionally require that from `beg` to `end` at least 10 e-folds pass:
            idx_end = np.argwhere(b._N - b._N[idx_beg] > 10).ravel()[0]
        if b.independent_variable == 't':
            p = PerturbationT(background=b, k=ki, vacuum=vacuum,
                              idx_beg=idx_beg, idx_end=idx_end, num_eval=num_eval)
        elif b.independent_variable == '_N':
            p = PerturbationN(background=b, k=ki, vacuum=vacuum,
                              idx_beg=idx_beg, idx_end=idx_end, num_eval=num_eval)
        else:
            raise NotImplementedError()
        oscode_sol = []
        for mode in [p.scalar, p.tensor]:
            for num in range(2):
                oscode_sol.append(pyoscode.solve(
                    ts=b.x[idx_beg:idx_end+1],
                    ti=b.x[idx_beg], tf=b.x[idx_end],
                    ws=np.log(mode.ms_frequency), logw=True,
                    gs=mode.ms_damping, logg=False,
                    x0=y0[2*num],
                    dx0=y0[2*num+1],
                    rtol=rtol, even_grid=even_grid,
                    t_eval=mode.x_eval,
                ))
        p.oscode_postprocessing(oscode_sol=oscode_sol, vacuum=vacuum)
        if ki < 1 and b.K == +1 and drop_closed_large_scales:
            p.scalar.P_s_RST = 1e-30
        for vac in vacuum:
            getattr(PPS, 'P_s_%s' % vac)[i] = getattr(p.scalar, 'P_s_%s' % vac)
            getattr(PPS, 'P_t_%s' % vac)[i] = getattr(p.tensor, 'P_t_%s' % vac)
    if return_pps:
        return PPS
    else:
        return p
