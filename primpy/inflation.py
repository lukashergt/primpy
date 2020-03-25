#!/usr/bin/env python
""":mod:`primpy.inflation`: general setup for equations for cosmic inflation."""
from abc import ABC
import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import derivative
from primpy.units import pi, c, a_B, mp_kg, lp_m, Mpc_m
from primpy.parameters import T_CMB, K_STAR
from primpy.equations import Equations


class InflationEquations(Equations, ABC):
    """Base class for inflation equations."""

    def __init__(self, K, potential):
        super(InflationEquations, self).__init__()
        self.K = K
        self.potential = potential

    def H(self, x, y):
        """Hubble parameter."""
        return np.sqrt(self.H2(x, y))

    def H2(self, x, y):
        """Hubble parameter squared."""
        raise NotImplementedError("Equations must define H2 method")

    def V(self, x, y):
        """Inflationary Potential."""
        return self.potential.V(self.phi(x, y))

    def dVdphi(self, x, y):
        """First derivative of inflationary potential."""
        return self.potential.dV(self.phi(x, y))

    def d2Vdphi2(self, x, y):
        """Second derivative of inflationary potential."""
        return self.potential.d2V(self.phi(x, y))

    def w(self, x, y):
        """Equation of state parameter."""
        raise NotImplementedError("Equations must define w method")

    def sol(self, sol, **kwargs):
        """Post-processing of `solve_ivp` solution."""
        sol = super(InflationEquations, self).sol(sol, **kwargs)
        sol.K = self.K
        sol.potential = self.potential
        sol.H = self.H(sol.x, sol.y)
        if not hasattr(sol, 'logaH'):
            sol.logaH = sol.N + np.log(sol.H)
        sol.w = self.w(sol.x, sol.y)
        sol.N_tot = sol.N_end - sol.N_beg
        sol.inflation_mask = (sol.N_beg <= sol.N) & (sol.N <= sol.N_end)

        def derive_a0(Omega_K0, h, delta_reh=None, w_reh=None):
            """Derive the scale factor today `a_0` either from reheating or from `Omega_K0`."""
            # derive a0 and Omega_K0 from reheating:
            if Omega_K0 is None:
                rho_r0_SI = a_B * T_CMB**4 / c**2
                rho_r0 = rho_r0_SI / mp_kg * lp_m**3
                # just from instant reheating:
                N0 = sol.N_end + np.log(3 / 2) / 4 + np.log(sol.V_end / rho_r0) / 4
                # additional term from general reheating:
                if delta_reh is not None and w_reh is not None:
                    N0 += (1 - 3 * w_reh) * delta_reh / 4
                sol.a0_lp = np.exp(N0)
                sol.a0_Mpc = sol.a0_lp * lp_m / Mpc_m
                sol.Omega_K0 = - sol.K * c**2 / (sol.a0_Mpc * 100e3 * h)**2
            # for flat universes the scale factor can be freely rescaled
            elif Omega_K0 == 0:
                assert sol.K == 0, \
                    ("The global geometry needs to match, "
                     "but Omega_K0=%s whereas K=%s." % (Omega_K0, sol.K))
                sol.Omega_K0 = Omega_K0
                sol.a0 = 1.
            # derive a0 from Omega_K0
            else:
                assert np.sign(Omega_K0) == -sol.K, \
                    ("The global geometry needs to match, "
                     "but Omega_K0=%s whereas K=%s." % (Omega_K0, sol.K))
                sol.Omega_K0 = Omega_K0
                sol.a0_Mpc = c / (100e3 * h) * np.sqrt(-sol.K / Omega_K0)
                sol.a0_lp = sol.a0_Mpc * Mpc_m / lp_m

        sol.derive_a0 = derive_a0

        def calibrate_a_flat_universe(N_star):
            """Calibrate the scale factor `a` for a flat universe using a given `N_star`."""
            # TODO: double check this function
            assert sol.K == 0
            sol.N_star = N_star  # number e-folds of inflation after horizon crossing
            sol.N_cross = sol.N_end - sol.N_star  # horizon crossing of pivot scale
            derive_a0(Omega_K0=0, h=None)

            # Calibrate aH=k using N_star at pivot scale K_STAR:
            N2logaH = interp1d(sol.N[sol.inflation_mask], sol.logaH[sol.inflation_mask])
            sol.logaH_star = N2logaH(sol.N_cross)

            sol.N_calib = sol.N + np.log(sol.a0) - sol.logaH_star + np.log(K_STAR / Mpc_m * lp_m)
            sol.a_calib = np.exp(sol.N_calib)

        sol.calibrate_a_flat_universe = calibrate_a_flat_universe

        def derive_comoving_hubble_horizon_flat(N_star):
            """Derive the comoving Hubble horizon `cHH`."""
            # for flat universes we first need to calibrate the scale factor:
            calibrate_a_flat_universe(N_star)
            sol.cHH_lp = sol.a0 / (sol.a_calib * sol.H)
            sol.cHH_Mpc = sol.cHH_lp * lp_m / Mpc_m

        def derive_comoving_hubble_horizon_curved(Omega_K0, h, delta_reh=None, w_reh=None):
            """Derive the comoving Hubble horizon `cHH`."""
            # for curved universes a0 can be derived from Omega_K0:
            derive_a0(Omega_K0=Omega_K0, h=h, delta_reh=delta_reh, w_reh=w_reh)
            sol.cHH_Mpc = np.exp(-sol.logaH) * sol.a0_Mpc
            sol.cHH_lp = np.exp(-sol.logaH) * sol.a0_lp
            sol.Omega_K = -sol.K * np.exp(-2 * sol.logaH)

        if sol.K == 0:
            sol.derive_comoving_hubble_horizon = derive_comoving_hubble_horizon_flat
        else:
            sol.derive_comoving_hubble_horizon = derive_comoving_hubble_horizon_curved

        def calibrate_wavenumber_flat(N_star, **interp1d_kwargs):
            """Calibrate wavenumber for flat universes, then derive approximate power spectra."""
            calibrate_a_flat_universe(N_star=N_star)

            sol.N_dagg = sol.N_tot - sol.N_star
            logaH = sol.logaH[sol.inflation_mask]
            sol.logk = np.log(K_STAR) + logaH - sol.logaH_star

            derive_approx_power(**interp1d_kwargs)

        def calibrate_wavenumber_curved(Omega_K0, h, delta_reh=None, w_reh=None,
                                        **interp1d_kwargs):
            """Calibrate wavenumber for curved universes, then derive approximate power spectra."""
            derive_a0(Omega_K0=Omega_K0, h=h, delta_reh=delta_reh, w_reh=w_reh)

            N = sol.N[sol.inflation_mask]
            logaH = sol.logaH[sol.inflation_mask]
            sol.logk = logaH - np.log(sol.a0_Mpc)
            if np.log(K_STAR) < np.min(sol.logk) or np.log(K_STAR) > np.max(sol.logk):
                sol.N_cross = np.nan
            else:
                logk, indices = np.unique(sol.logk, return_index=True)
                logk2N = interp1d(logk, N[indices])
                sol.N_cross = logk2N(np.log(K_STAR))
            sol.N_dagg = sol.N_cross - sol.N_beg
            sol.N_star = sol.N_end - sol.N_cross

            derive_approx_power(**interp1d_kwargs)

        def derive_approx_power(**interp1d_kwargs):
            """Derive the approximate primordial power spectra for scalar and tensor modes."""
            H = sol.H[sol.inflation_mask]
            dphidt = sol.dphidt[sol.inflation_mask]
            sol.P_scalar_approx = (H**2 / (2 * pi * dphidt))**2
            sol.P_tensor_approx = 2 * (H / pi)**2

            logk, indices = np.unique(sol.logk, return_index=True)

            bounds_error = interp1d_kwargs.pop('bounds_error', False)
            fill_value = interp1d_kwargs.pop('fill_value', 0)
            kind = interp1d_kwargs.pop('kind', 'cubic')
            sol.logk2P_scalar = interp1d(logk, sol.P_scalar_approx[indices],
                                         bounds_error=bounds_error,
                                         fill_value=fill_value,
                                         kind=kind)
            sol.logk2P_tensor = interp1d(logk, sol.P_tensor_approx[indices],
                                         bounds_error=bounds_error,
                                         fill_value=fill_value,
                                         kind=kind)
            derive_approx_ns()
            derive_approx_nrun()
            derive_approx_r()
            derive_approx_As()

        def derive_approx_ns():
            """Derive the spectral index `n_s` from `P_s_approx`."""
            def logP(logk):
                """Help function for scipy's derivative."""
                return np.log(sol.P_s_approx(np.exp(logk)))

            sol.n_s = 1 + derivative(func=logP, x0=np.log(K_STAR),
                                     dx=np.log(K_STAR)/10, n=1, order=9)

        def derive_approx_nrun(dx=np.log(K_STAR)/10., order=9):
            """Derive the running of the spectral index `n_run` from `P_s_approx`."""
            def logP(logk):
                """Help function for scipy's derivative."""
                return np.log(sol.P_s_approx(np.exp(logk)))

            sol.n_run = derivative(func=logP, x0=np.log(K_STAR), dx=dx, n=2,
                                   order=order)
            return sol.nrun

        def derive_approx_r():
            """Derive the tensor-to-scalar ratio `r` from `P_s_approx`."""
            sol.r = sol.P_t_approx(K_STAR) / sol.P_s_approx(K_STAR)

        def derive_approx_As():
            """Derive the amplitude `A_s` from `P_s_approx`."""
            sol.A_s = sol.P_s_approx(K_STAR)

        if self.K == 0:
            sol.derive_approx_power = calibrate_wavenumber_flat
        else:
            sol.derive_approx_power = calibrate_wavenumber_curved

        def P_s_approx(k):
            """Slow-roll approximation for the primordial power spectrum for scalar modes."""
            return sol.logk2P_scalar(np.log(k))

        def P_t_approx(k):
            """Slow-roll approximation for the primordial power spectrum for tensor modes."""
            return sol.logk2P_tensor(np.log(k))

        sol.P_s_approx = P_s_approx
        sol.P_t_approx = P_t_approx

        return sol