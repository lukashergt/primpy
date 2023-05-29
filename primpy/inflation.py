#!/usr/bin/env python
""":mod:`primpy.inflation`: general setup for equations for cosmic inflation."""
from warnings import warn
from abc import ABC
import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from primpy.exceptionhandling import CollapseWarning, InflationStartWarning, InflationEndWarning
from primpy.units import pi, c, lp_m, Mpc_m
from primpy.parameters import K_STAR, rho_r0_mp_ilp3
from primpy.equations import Equations


class InflationEquations(Equations, ABC):
    """Base class for inflation equations."""

    def __init__(self, K, potential, verbose=False):
        super(InflationEquations, self).__init__()
        self.vwarn = warn if verbose else lambda *a, **k: None
        self.K = K
        self.potential = potential

    def H(self, x, y):
        """Hubble parameter."""
        return np.sqrt(self.H2(x, y))

    def H2(self, x, y):
        """Hubble parameter squared."""
        raise NotImplementedError("Equations must define H2 method.")

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
        raise NotImplementedError("Equations must define w method.")

    def inflating(self, x, y):
        """Inflation diagnostic for event tracking."""
        raise NotImplementedError("Equations must define inflating method.")

    def postprocessing_inflation_start(self, sol):
        """Extract starting point of inflation from event tracking."""
        sol.N_beg = np.nan
        # Case 0: Universe has collapsed
        if 'Collapse' in sol.N_events and sol.N_events['Collapse'].size > 0:
            self.vwarn(CollapseWarning(""))
        # Case 1: inflating from the start
        elif self.inflating(sol.x[0], sol.y[:, 0]) >= 0 or sol.w[0] <= -1/3:
            sol.N_beg = sol.N[0]
        # Case 2: there is a transition from non-inflating to inflating
        elif ('Inflation_dir1_term0' in sol.N_events and
              np.size(sol.N_events['Inflation_dir1_term0']) > 0):
            sol.N_beg = sol.N_events['Inflation_dir1_term0'][0]
        else:
            self.vwarn(InflationStartWarning("", events=sol.N_events))

    def postprocessing_inflation_end(self, sol):
        """Extract end point of inflation from event tracking."""
        sol.N_end = np.nan
        sol.phi_end = np.nan
        sol.H_end = np.nan
        sol.V_end = np.nan
        # end of inflation is first transition from inflating to non-inflating
        for key in ['Inflation_dir-1_term1', 'Inflation_dir-1_term0']:
            if key in sol.N_events and sol.N_events[key].size > 0:
                sol.N_end = sol.N_events[key][0]
                sol.phi_end = sol.phi_events[key][0]
                sol.H_end = self.H(sol.x_events[key][0], sol.y_events[key][0])
                break
        if np.isfinite(sol.phi_end):
            sol.V_end = self.potential.V(sol.phi_end)
        else:
            self.vwarn(InflationEndWarning("", events=sol.N_events, sol=sol))

    def sol(self, sol, **kwargs):
        """Post-processing of :func:`scipy.integrate.solve_ivp` solution."""
        sol = super(InflationEquations, self).sol(sol, **kwargs)
        sol.w = self.w(sol.x, sol.y)
        self.postprocessing_inflation_start(sol)
        self.postprocessing_inflation_end(sol)
        sol.K = self.K
        sol.potential = self.potential
        sol.H = self.H(sol.x, sol.y)
        sol.logaH = sol.N + np.log(sol.H)
        sol.Omega_K = -sol.K * np.exp(-2 * sol.logaH)
        sol.N_tot = sol.N_end - sol.N_beg
        if np.isfinite(sol.N_beg) and np.isfinite(sol.N_end):
            sol.inflation_mask = (sol.N_beg <= sol.N) & (sol.N <= sol.N_end)

        def derive_a0(Omega_K0, h, delta_reh=None, w_reh=None):
            """Derive the scale factor today `a_0` either from reheating or from `Omega_K0`."""
            # derive a0 and Omega_K0 from reheating:
            if Omega_K0 is None:
                # just from instant reheating:
                N0 = sol.N_end + np.log(3 / 2) / 4 + np.log(sol.V_end / rho_r0_mp_ilp3) / 4
                # additional term from general reheating:
                if delta_reh is not None and w_reh is not None:
                    N0 += (1 - 3 * w_reh) * delta_reh / 4
                sol.a0_lp = np.exp(N0)
                sol.a0_Mpc = sol.a0_lp * lp_m / Mpc_m
                sol.Omega_K0 = - sol.K * c**2 / (sol.a0_Mpc * 100e3 * h)**2
            # for flat universes the scale factor can be freely rescaled
            elif Omega_K0 == 0:
                assert sol.K == 0, ("The global geometry needs to match, "
                                    "but Omega_K0=%s whereas K=%s." % (Omega_K0, sol.K))
                sol.Omega_K0 = Omega_K0
                sol.a0 = 1.
            # derive a0 from Omega_K0
            else:
                assert np.sign(Omega_K0) == -sol.K, ("The global geometry needs to match, "
                                                     "but Omega_K0=%s whereas K=%s."
                                                     % (Omega_K0, sol.K))
                sol.Omega_K0 = Omega_K0
                sol.a0_Mpc = c / (100e3 * h) * np.sqrt(-sol.K / Omega_K0)
                sol.a0_lp = sol.a0_Mpc * Mpc_m / lp_m

        sol.derive_a0 = derive_a0

        def calibrate_a_flat_universe(N_star, logaH_star=None):
            """Calibrate the scale factor `a` for a flat universe using a given `N_star`."""
            # TODO: double check this function
            assert sol.K == 0
            derive_a0(Omega_K0=0, h=None)
            if logaH_star is None:
                sol.N_star = N_star  # number e-folds of inflation after horizon crossing
                sol.N_cross = sol.N_end - sol.N_star  # horizon crossing of pivot scale
                # Calibrate aH=k using N_star at pivot scale K_STAR:
                N2logaH = interp1d(sol.N[sol.inflation_mask], sol.logaH[sol.inflation_mask])
                sol.logaH_star = N2logaH(sol.N_cross)
            else:  # allows manual override, e.g. when integrating backwards without any N_cross
                sol.logaH_star = logaH_star

            sol.N_calib = sol.N + np.log(sol.a0) - sol.logaH_star + np.log(K_STAR / Mpc_m * lp_m)
            sol.a_calib = np.exp(sol.N_calib)
            sol.a0_Mpc = np.exp(sol.logaH_star) / K_STAR

        def derive_comoving_hubble_horizon_flat(N_star, logaH_star=None):
            """Derive the comoving Hubble horizon `cHH`."""
            # for flat universes we first need to calibrate the scale factor:
            calibrate_a_flat_universe(N_star, logaH_star)
            sol.cHH_lp = sol.a0 / (sol.a_calib * sol.H)
            sol.cHH_Mpc = sol.cHH_lp * lp_m / Mpc_m

        def derive_comoving_hubble_horizon_curved(Omega_K0, h, delta_reh=None, w_reh=None):
            """Derive the comoving Hubble horizon `cHH`."""
            # for curved universes a0 can be derived from Omega_K0:
            derive_a0(Omega_K0=Omega_K0, h=h, delta_reh=delta_reh, w_reh=w_reh)
            sol.cHH_Mpc = np.exp(-sol.logaH) * sol.a0_Mpc
            sol.cHH_lp = np.exp(-sol.logaH) * sol.a0_lp
            sol.cHH_end_Mpc = sol.a0_Mpc * np.exp(-sol.N_end) / sol.H_end
            sol.cHH_end_lp = sol.a0_lp * np.exp(-sol.N_end) / sol.H_end
            sol.log_cHH_end_Mpc = np.log(sol.a0_Mpc) - sol.N_end - np.log(sol.H_end)
            sol.log_cHH_end_lp = np.log(sol.a0_lp) - sol.N_end - np.log(sol.H_end)

        if sol.K == 0:
            sol.derive_comoving_hubble_horizon = derive_comoving_hubble_horizon_flat
        else:
            sol.derive_comoving_hubble_horizon = derive_comoving_hubble_horizon_curved

        def calibrate_wavenumber_flat(N_star, logaH_star=None, **interp1d_kwargs):
            """Calibrate wavenumber for flat universes, then derive approximate power spectra."""
            calibrate_a_flat_universe(N_star, logaH_star)

            sol.N_dagg = sol.N_tot - sol.N_star
            logaH = sol.logaH[sol.inflation_mask]
            sol.logk = np.log(K_STAR) + logaH - sol.logaH_star
            sol.k_iMpc = np.exp(sol.logk)
            sol.k_comoving = sol.k_iMpc * sol.a0_Mpc

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
            if hasattr(sol, 'dphidt'):
                dphidt = sol.dphidt[sol.inflation_mask]
            else:
                dphidt = H * sol.dphidN[sol.inflation_mask]
            sol.P_scalar_approx = (H**2 / (2 * pi * dphidt))**2
            sol.P_tensor_approx = 2 * (H / pi)**2

            logk, indices = np.unique(sol.logk, return_index=True)
            spline_order = interp1d_kwargs.pop('k', 3)
            extrapolate = interp1d_kwargs.pop('ext', 'zeros')
            sol.logk2logP_s = InterpolatedUnivariateSpline(logk,
                                                           np.log(sol.P_scalar_approx[indices]),
                                                           k=spline_order, ext=extrapolate,
                                                           **interp1d_kwargs)
            sol.logk2logP_t = InterpolatedUnivariateSpline(logk,
                                                           np.log(sol.P_tensor_approx[indices]),
                                                           k=spline_order, ext=extrapolate,
                                                           **interp1d_kwargs)
            if sol.logk[0] < np.log(K_STAR) < sol.logk[-1]:
                dlogPdlogk_s = sol.logk2logP_s.derivatives(np.log(K_STAR))
                dlogPdlogk_t = sol.logk2logP_t.derivatives(np.log(K_STAR))
                sol.A_s = np.exp(dlogPdlogk_s[0])
                sol.n_s = 1 + dlogPdlogk_s[1]
                sol.n_run = dlogPdlogk_s[2]
                sol.n_runrun = dlogPdlogk_s[3]
                sol.A_t = np.exp(dlogPdlogk_t[0])
                sol.n_t = dlogPdlogk_t[1]
                sol.r = sol.A_t / sol.A_s
            else:
                sol.A_s = np.nan
                sol.n_s = np.nan
                sol.n_run = np.nan
                sol.n_runrun = np.nan
                sol.A_t = np.nan
                sol.n_t = np.nan
                sol.r = np.nan

        if self.K == 0:
            sol.derive_approx_power = calibrate_wavenumber_flat
        else:
            sol.derive_approx_power = calibrate_wavenumber_curved

        def P_s_approx(k):
            """Slow-roll approximation for the primordial power spectrum for scalar modes."""
            return np.exp(sol.logk2logP_s(np.log(k)))

        def P_t_approx(k):
            """Slow-roll approximation for the primordial power spectrum for tensor modes."""
            return np.exp(sol.logk2logP_t(np.log(k)))

        sol.P_s_approx = P_s_approx
        sol.P_t_approx = P_t_approx

        return sol
