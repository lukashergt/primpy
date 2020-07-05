#!/usr/bin/env python
""":mod:`primpy.bigbang`: general setup for equations for standard Big Bang cosmology."""
import numpy as np
from scipy import integrate
from primpy.units import pi, G, tp_s, lp_m, Mpc_m
from primpy.parameters import rho_r0_kg_im3


def get_H0(h, units='planck'):
    """Get present-day Hubble parameter from little Hubble `h`."""
    if units == 'planck':
        return h * 100e3 / Mpc_m * tp_s  # in reduced Planck units
    elif units == 'H0':
        return h * 100  # in conventional Hubble parameter units
    elif units == 'SI':
        return h * 100e3 / Mpc_m  # in SI units, i.e. s^-1
    else:
        raise NotImplementedError("%s not implemented for units, please choose "
                                  "one of {'planck', 'H0', 'SI'}." % units)


def get_a0(h, Omega_K0, units='planck'):
    """Get present-day scale factor from curvature density parameter."""
    if Omega_K0 == 0:
        return 1
    H0 = get_H0(h, units='planck')
    K = -np.sign(Omega_K0)
    a0 = np.sqrt(-K / Omega_K0) / H0
    if units == 'planck':
        return a0  # in reduced Planck units, i.e. lp
    elif units == 'Mpc':
        return a0 * lp_m / Mpc_m  # in Mpc
    elif units == 'SI':
        return a0 * lp_m  # in SI units, i.e. m
    else:
        raise NotImplementedError("%s not implemented for units, please choose "
                                  "one of {'planck', 'Mpc', 'SI'}." % units)


def get_rho_crit_kg_im3(h):
    """Get present-day critical density from little Hubble `h`."""
    H0_is = get_H0(h=h, units='SI')
    rho_crit_kg_im3 = 3 * H0_is**2 / (8 * pi * G)
    return rho_crit_kg_im3


def get_Omega_r0(h):
    """Get present-day radiation density parameter from little Hubble `h`."""
    rho_crit_kg_im3 = get_rho_crit_kg_im3(h=h)
    Omega_r0 = rho_r0_kg_im3 / rho_crit_kg_im3
    return Omega_r0


def Hubble_parameter(N, Omega_m0, Omega_K0, h):
    """Hubble parameter (in reduced Planck units) at N=ln(a) during standard Big Bang.

    Parameters
    ----------
        N : float, np.ndarray
            e-folds of the scale factor N=ln(a) during standard Big Bang
            evolution, where the scale factor would be given in reduced Planck
            units (same as output from primpy).
        Omega_m0 : float
            matter density parameter today
        Omega_K0 : float
            curvature density parameter today
        h : float
            dimensionless Hubble parameter today, "little h"

    Omega_r0 is derived from the Hubble parameter using Planck's law.
    Omega_L0 is derived from the other density parameters to sum to one.

    Returns
    -------
        H : float
            Hubble parameter during standard Big Bang evolution of the Universe.
            In reduced Planck units [tp^-1].
    """
    a = np.exp(N)
    H0 = get_H0(h=h, units='planck')  # in reduced Planck units
    Omega_r0 = get_Omega_r0(h=h)
    Omega_L0 = 1 - Omega_r0 - Omega_m0 - Omega_K0
    if Omega_L0 > no_Big_Bang_line(Omega_m0=Omega_m0):
        raise Exception("no Big Bang for Omega_m0=%g, Omega_L0=%g" % (Omega_m0, Omega_L0))
    elif Omega_L0 < expand_recollapse_line(Omega_m0=Omega_m0):
        raise Exception("Universe recollapses for Omega_m0=%g, Omega_L0=%g" % (Omega_m0, Omega_L0))
    a0 = get_a0(h=h, Omega_K0=Omega_K0, units='planck')
    H = H0 * np.sqrt(Omega_r0 * (a0 / a)**4 +
                     Omega_m0 * (a0 / a)**3 +
                     Omega_K0 * (a0 / a)**2 +
                     Omega_L0)
    return H


def no_Big_Bang_line(Omega_m0):
    """Return `Omega_L0` for dividing line between universes with/without Big Bang.

    Parameters
    ----------
        Omega_m0 : float
            matter density parameter today

    Returns
    -------
        Omega_L0 : float
            Density parameter of cosmological constant `Lambda` along the
            dividing line between a Big Bang evolution (for smaller Omega_L0)
            and universes without a Big Bang (for larger Omega_L0).
    """
    if Omega_m0 == 0:
        return 1
    if 0 < Omega_m0 <= 0.5:
        return 4 * Omega_m0 * np.cosh(np.arccosh((1 - Omega_m0) / Omega_m0) / 3)**3
    elif 0.5 <= Omega_m0:
        return 4 * Omega_m0 * np.cos(np.arccos((1 - Omega_m0) / Omega_m0) / 3)**3
    else:
        raise ValueError("Matter density can't be negative but, Omega_m0=%g" % Omega_m0)


def expand_recollapse_line(Omega_m0):
    """Return `Omega_L0` for dividing line between expanding/recollapsing universes.

    Parameters
    ----------
        Omega_m0 : float
            matter density parameter today

    Returns
    -------
        Omega_L0 : float
            Density parameter of cosmological constant `Lambda` along the
            dividing line between expanding (for larger Omega_L0) and
            recollapsing (for smaller Omega_L0) universes.
    """
    if 0 <= Omega_m0 < 1:
        return 0
    elif 1 <= Omega_m0:
        return 4 * Omega_m0 * np.cos(np.arccos((1 - Omega_m0) / Omega_m0) / 3 + 4*pi/3)**3
    else:
        raise ValueError("Matter density can't be negative but, Omega_m0=%g" % Omega_m0)


def comoving_Hubble_horizon(N, Omega_m0, Omega_K0, h, units='planck'):
    """Comoving Hubble horizon at N=ln(a) during standard Big Bang.

    Parameters
    ----------
        N : float, np.ndarray
            e-folds of the scale factor N=ln(a) during standard Big Bang
            evolution, where the scale factor would be given in reduced Planck
            units (same as output from primpy).
        Omega_m0 : float
            matter density parameter today
        Omega_K0 : float
            curvature density parameter today
        h : float
            dimensionless Hubble parameter today, "little h"
        units : str
            Output units, can be any of {'planck', 'Mpc', 'SI'} returning
            units of `lp`, `Mpc` or `m` respectively.

    Omega_r0 is derived from the Hubble parameter using Planck's law.
    Omega_L0 is derived from the other density parameters to sum to one.

    Returns
    -------
        cHH : float
            Comoving Hubble horizon during standard Big Bang evolution of the Universe.

    """
    a0 = get_a0(h=h, Omega_K0=Omega_K0, units=units)
    a = np.exp(N)
    H = Hubble_parameter(N=N, Omega_m0=Omega_m0, Omega_K0=Omega_K0, h=h)
    return a0 / (a * H)


def conformal_time(N_start, N, Omega_m0, Omega_K0, h):
    """Conformal time during standard Big Bang evolution from N_start to N.

    Parameters
    ----------
        N_start : float
            e-folds of the scale factor N=ln(a) during standard Big Bang
            evolution at lower integration limit (e.g. at end of inflation),
            where the scale factor would be given in reduced Planck units
            (same as output from primpy).
        N : float, np.ndarray
            e-folds of the scale factor N=ln(a) during standard Big Bang
            evolution at upper integration limit (e.g. at end of inflation),
            where the scale factor would be given in reduced Planck units
            (same as output from primpy).
        Omega_m0 : float
            matter density parameter today
        Omega_K0 : float
            curvature density parameter today
        h : float
            dimensionless Hubble parameter today, "little h"

    Omega_r0 is derived from the Hubble parameter using Planck's law and from N_eff.
    Omega_L0 is derived from the other density parameters to sum to one.

    Returns
    -------
        eta : float, np.ndarray
            conformal time passing between `a_start` and `a`
            during standard Big Bang evolution of the Universe.
            Same shape as `N`.
    """
    if isinstance(N, np.ndarray):
        return np.array([conformal_time(N_start=N_start, N=n, Omega_m0=Omega_m0,
                                        Omega_K0=Omega_K0, h=h)[0] for n in N])
    elif isinstance(N, float) or isinstance(N, int):
        def integrand(n):
            return np.exp(-n) / Hubble_parameter(N=n, Omega_m0=Omega_m0, h=h, Omega_K0=Omega_K0)
        eta = integrate.quad(func=integrand, a=N_start, b=N)
        return eta
    else:
        raise Exception("`N` needs to be either float or np.ndarray of floats, "
                        "but is type(N)=%s" % type(N))
