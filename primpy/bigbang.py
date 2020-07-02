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
        NotImplementedError("%s not implemented for units, please choose one of "
                            "{'planck', 'H0', 'SI'}." % units)


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
        NotImplementedError("%s not implemented for units, please choose one of "
                            "{'planck', 'Mpc', 'SI'}." % units)


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
    ==========
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
    =======
        H : float
            Hubble parameter during standard Big Bang evolution of the Universe.
            In reduced Planck units [tp^-1].
    """
    a = np.exp(N)
    H0 = get_H0(h=h, units='tp^-1')  # in reduced Planck units
    Omega_r0 = get_Omega_r0(h=h)
    Omega_L0 = 1 - Omega_r0 - Omega_m0 - Omega_K0
    a0 = get_a0(h=h, Omega_K0=Omega_K0, units='planck')
    H = H0 * np.sqrt(Omega_r0 * (a0 / a)**4 +
                     Omega_m0 * (a0 / a)**3 +
                     Omega_K0 * (a0 / a)**2 +
                     Omega_L0)
    return H


def comoving_Hubble_horizon(N, Omega_m0, Omega_K0, h, units='planck'):
    """Comoving Hubble horizon at N=ln(a) during standard Big Bang.

    Parameters
    ==========
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
    =======
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
    ==========
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
    =======
        eta : float
            conformal time passing between `a_start` and `a`
            during standard Big Bang evolution of the Universe.
    """

    def integrand(n):
        return np.exp(-n) / Hubble_parameter(N=n, Omega_m0=Omega_m0, h=h, Omega_K0=Omega_K0)
    eta = integrate.quad(func=integrand, a=N_start, b=N)
    return eta
