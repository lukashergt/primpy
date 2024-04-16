#!/usr/bin/env python
""":mod:`primpy.inflation`: general setup for equations for cosmic inflation."""
from warnings import warn
from abc import ABC
import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from primpy.exceptionhandling import CollapseWarning, InflationStartWarning, InflationEndWarning
from primpy.exceptionhandling import InsufficientInflationError
from primpy.units import pi, c, lp_m, Mpc_m, mp_GeV, lp_iGeV
from primpy.parameters import K_STAR, K_STAR_lp, T_CMB_Tp, g0
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
        sol._N_beg = np.nan
        sol.H_beg = np.nan
        # Case 0: Universe has collapsed
        if 'Collapse' in sol._N_events and sol._N_events['Collapse'].size > 0:
            self.vwarn(CollapseWarning(""))
        # Case 1: inflating from the start
        elif self.inflating(sol.x[0], sol.y[:, 0]) >= 0 or sol.w[0] <= -1/3:
            sol._N_beg = sol._N[0]
            sol.H_beg = self.H(sol.x[0], sol.y[:, 0])
        # Case 2: there is a transition from non-inflating to inflating
        elif ('Inflation_dir1_term0' in sol._N_events and
              np.size(sol._N_events['Inflation_dir1_term0']) > 0):
            sol._N_beg = sol._N_events['Inflation_dir1_term0'][0]
            sol.H_beg = self.H(sol.x_events['Inflation_dir1_term0'][0],
                               sol.y_events['Inflation_dir1_term0'][0])
        else:
            self.vwarn(InflationStartWarning("", events=sol._N_events))
        sol._logaH_beg = sol._N_beg + np.log(sol.H_beg)

    def postprocessing_inflation_end(self, sol):
        """Extract end point of inflation from event tracking."""
        sol._N_end = np.nan
        sol._logaH_end = np.nan
        sol.phi_end = np.nan
        sol.H_end = np.nan
        sol.V_end = np.nan
        # end of inflation is first transition from inflating to non-inflating
        for key in ['Inflation_dir-1_term1', 'Inflation_dir-1_term0']:
            if key in sol._N_events and sol._N_events[key].size > 0:
                sol._N_end = sol._N_events[key][0]
                sol.phi_end = sol.phi_events[key][0]
                sol.H_end = self.H(sol.x_events[key][0], sol.y_events[key][0])
                sol._logaH_end = sol._N_end + np.log(sol.H_end)
                break
        if np.isfinite(sol.phi_end):
            sol.V_end = self.potential.V(sol.phi_end)
        else:
            self.vwarn(InflationEndWarning("", events=sol._N_events, sol=sol))

    def sol(self, sol, **kwargs):
        """Post-processing of :func:`scipy.integrate.solve_ivp` solution."""
        sol = super(InflationEquations, self).sol(sol, **kwargs)
        sol.w = self.w(sol.x, sol.y)
        self.postprocessing_inflation_start(sol)
        self.postprocessing_inflation_end(sol)
        sol.K = self.K
        sol.potential = self.potential
        sol.H = self.H(sol.x, sol.y)
        sol._logaH = sol._N + np.log(sol.H)
        sol.Omega_K = -sol.K * np.exp(-2 * sol._logaH)
        sol.N_tot = sol._N_end - sol._N_beg
        if np.isfinite(sol._N_beg) and np.isfinite(sol._N_end):
            sol.inflation_mask = (sol._N_beg <= sol._N) & (sol._N <= sol._N_end)

        def calibrate_scale_factor(
                calibration_method='N_star' if self.K == 0 else 'Omega_K0',
                Omega_K0=None, h=None,                                   # for curved universes
                N_star=None, logaH_star=None,                            # for flat universes
                delta_reh=None, w_reh=None, rho_reh_GeV=None, g_th=1e2  # for reheating
        ):
            """Calibrate the scale factor `a` for flat or curved universes or from reheating.

            Computes the following attributes:
            - `a0`: scale factor today (in Planck units), set to 1 for flat universes.
            - `N0`: e-fold number today, i.e. `N0 = ln(a0)`.
            - `N`: independent e-folds variable calibrated to match `aH` to `k`.
            - `N_star`: e-folds of inflation after horizon crossing of pivot scale `K_STAR`.
            - `N_dagg`: e-folds of inflation before horizon crossing of pivot scale `K_STAR`.
            - `k_iMpc`: wavenumber in inverse Mpc.

            Parameters
            ----------
            calibration_method : str
                Method to calibrate the scale factor. Choose from:

                    - flat universes: 'N_star' (default) or 'reheating'
                    - curved universes: 'Omega_K0' (default) or 'reheating'

            Omega_K0 : float
                Curvature density parameter today. Required for ``calibration_method='Omega_K0'``.

            h : float
                Hubble parameter today. Required for ``Omega_K0`` and ``reheating`` calibration.

            N_star : float
                Number of e-folds of inflation after horizon crossing of pivot scale `K_STAR`.
                Required for ``calibration_method='N_star'``.

            logaH_star : float, optional
                Optional calibration factor for flat universes to match `aH` to `k` that
                circumvents the full computation of the calibration. This is needed e.g. when
                integrating backwards in time where `_N_end` and `_N_cross` are not available.
                When provided, all other input parameters are ignored.

            delta_reh : float, optional
                Number of e-folds during reheating, used for a general reheating scenario with
                ``calibration_method='reheating'``. By default, this is assumed to be zero,
                corresponding to instant reheating.

            w_reh : float, optional
                Equation of state parameter during reheating, used for a general reheating scenario
                with ``calibration_method='reheating'``. By default, this is assumed to be 1/3,
                corresponding to instant reheating.

            rho_reh_GeV : float, optional
                Energy density at the end of reheating in GeV (note that this is units of GeV not
                GeV^4, so this is really the fourth root of the energy density `rho_reh^(1/4)`),
                used to derive reheating parameters from curvature for
                ``calibration_method='Omega_K0'``.

            g_th : float, default: 100
                Number of relativistic degrees of freedom at the end of reheating, used in
                reheating calculations making use of entropy conservation.

            """
            if self.K == 0:  # flat universe
                if Omega_K0 is not None and Omega_K0 != 0.0:
                    raise ValueError(f"For flat universes Omega_K0 must be 0, but got "
                                     f"Omega_K0={Omega_K0}.")
                sol.a0 = 1
                sol.N0 = 0
                sol.Omega_K0 = 0

                if logaH_star is None:
                    if calibration_method == 'N_star':  # derive _N_cross from N_star
                        if N_star is None or N_star <= 0:
                            raise ValueError(f"For calibration_method='N_star' N_star>0 must be "
                                             f"given, but got N_star={N_star}.")
                        if N_star > sol.N_tot:
                            raise InsufficientInflationError(
                                f"The total number of e-folds of inflation N_tot={sol.N_tot} is "
                                f"smaller than the requested number of e-folds after horizon "
                                f"crossing N_star={N_star}."
                            )
                        sol.N_star = N_star
                        sol._N_cross = sol._N_end - sol.N_star
                        N2logaH = interp1d(sol._N[sol.inflation_mask],
                                           sol._logaH[sol.inflation_mask])
                        sol._logaH_star = N2logaH(sol._N_cross)
                        sol.delta_N_calib = sol.N0 - sol._logaH_star + np.log(K_STAR_lp)
                        if rho_reh_GeV is None:
                            sol.rho_reh_GeV = np.nan
                            sol._N_reh = np.nan
                            sol.w_reh = np.nan
                            sol.delta_reh = np.nan
                        else:
                            sol.rho_reh_GeV = rho_reh_GeV
                            sol.rho_reh_mp4 = rho_reh_GeV**4 / mp_GeV * lp_iGeV**3
                            sol.N_reh = (sol.N0
                                         - np.log((45/pi**2)**(1/4) * g0**(-1/3))
                                         - np.log(g_th) / 12
                                         + np.log(3/2 * T_CMB_Tp**4 / sol.rho_reh_mp4) / 4)
                            sol._N_reh = sol.N_reh - sol.delta_N_calib
                            sol.delta_reh = sol._N_reh - sol._N_end
                            sol.w_reh = np.log(3/2*sol.V_end/sol.rho_reh_mp4)/(3*sol.delta_reh) - 1

                    elif calibration_method == 'reheating':  # derive _N_cross from reheating
                        sol.N_end = (sol.N0
                                     - np.log((45/pi**2)**(1/4) * g0**(-1/3))
                                     - np.log(g_th) / 12
                                     - np.log(sol.V_end/T_CMB_Tp**4)/4)
                        if ((w_reh is None and delta_reh is None)
                                or delta_reh == 0 or w_reh == 1/3):
                            # assume instant reheating
                            sol.delta_reh = 0
                            sol.w_reh = np.nan
                            sol.rho_reh_mp4 = 3/2 * sol.V_end
                            sol.rho_reh_GeV = (sol.rho_reh_mp4 * mp_GeV / lp_iGeV**3)**(1/4)
                        elif w_reh is not None and delta_reh is not None:
                            if delta_reh < 0 or w_reh < -1/3:
                                raise ValueError(f"delta_reh must be positive (end of reheating "
                                                 f"must be after end of inflation) and w_reh must "
                                                 f"be greater than -1/3 (reheating by definition "
                                                 f"happens after the end of inflation, but "
                                                 f"w_reh<-1/3 is inflating), but got "
                                                 f"delta_reh={delta_reh} and w_reh={w_reh}.")
                            sol.w_reh = w_reh
                            sol.delta_reh = delta_reh
                            sol.N_end -= 3/4 * (1/3 - w_reh) * delta_reh
                            sol.rho_reh_mp4 = 3/2 * sol.V_end * np.exp(-3 * (1+w_reh) * delta_reh)
                            sol.rho_reh_GeV = (sol.rho_reh_mp4 * mp_GeV / lp_iGeV**3)**(1/4)
                        elif ((w_reh is None and delta_reh is not None) or
                              (w_reh is not None and delta_reh is None)):
                            raise ValueError(f"Both w_reh and delta_reh must be given for "
                                             f"reheating (or set both to None for instant "
                                             f"reheating), but got w_reh={w_reh} and "
                                             f"delta_reh={delta_reh}.")
                        sol.delta_N_calib = sol.N_end - sol._N_end
                        sol._logaH_star = np.log(K_STAR_lp) - sol.delta_N_calib
                        logaH = sol._logaH[sol.inflation_mask] + sol.delta_N_calib
                        logaH2N = interp1d(logaH, sol._N[sol.inflation_mask])
                        if sol._logaH_star < sol._logaH_beg or sol._logaH_star > sol._logaH_end:
                            raise InsufficientInflationError(
                                f"Pivot scale log(K_STAR)={np.log(K_STAR_lp)} is not within the "
                                f"range of the comoving Hubble horizon during inflation: "
                                f"logaH_beg={sol._logaH_beg+sol.delta_N_calib}, "
                                f"logaH_end={sol._logaH_end+sol.delta_N_calib}."
                            )
                        else:
                            sol._N_cross = logaH2N(np.log(K_STAR_lp))

                        sol.N_star = sol._N_end - sol._N_cross
                        sol._N_reh = sol._N_end + sol.delta_reh

                    else:  # only 'N_star' and 'reheating' as calibration methods for K==0
                        raise NotImplementedError(
                            f"Unknown calibration_method={calibration_method}, choose from "
                            f"'N_star' or 'reheating' for flat universes."
                        )

                else:  # allows manual override, e.g. when integrating backwards without _N_cross
                    if N_star is None or N_star <= 0:
                        raise ValueError(f"To circumvent the calibration by providing logaH_star, "
                                         f"you nonetheless need to provide N_star>0, but got "
                                         f"N_star={N_star}.")
                    sol.N_star = N_star
                    sol._N_cross = sol._N_end - sol.N_star
                    sol._logaH_star = logaH_star
                sol.a0_Mpc = np.exp(sol._logaH_star) / K_STAR
                sol.logk = sol._logaH[sol.inflation_mask] + np.log(K_STAR) - sol._logaH_star

            else:  # curved universe
                if h is None or h <= 0:
                    raise ValueError(f"To calibrate curved universes little h>0 must be provided, "
                                     f"but got h={h}.")
                sol.delta_N_calib = 0  # already calibrated through initial curvature Omega_Ki

                if calibration_method == 'Omega_K0':  # derive a0 from curvature using Omega_K0
                    if Omega_K0 is None or Omega_K0 == 0:
                        raise ValueError(f"For calibration_method='Omega_K0', Omega_K0!=0 must be "
                                         f"given, but got Omega_K0={Omega_K0}.")
                    elif np.sign(Omega_K0) != -sol.K:
                        raise ValueError(f"The global geometry needs to match, but "
                                         f"Omega_K0={Omega_K0} whereas K={sol.K}.")
                    sol.Omega_K0 = Omega_K0
                    sol.a0_Mpc = c / (h * 100 * 1e3) * np.sqrt(-sol.K / Omega_K0)
                    sol.a0 = sol.a0_Mpc * Mpc_m / lp_m
                    sol.N0 = np.log(sol.a0)
                    if rho_reh_GeV is None:
                        sol.rho_reh_GeV = np.nan
                        sol._N_reh = np.nan
                        sol.w_reh = np.nan
                        sol.delta_reh = np.nan
                    else:
                        sol.rho_reh_GeV = rho_reh_GeV
                        sol.rho_reh_mp4 = rho_reh_GeV**4 / mp_GeV * lp_iGeV**3
                        sol._N_reh = (sol.N0
                                      - np.log((45/pi**2)**(1/4) * g0**(-1/3))
                                      - np.log(g_th) / 12
                                      + np.log(3/2 * T_CMB_Tp**4 / sol.rho_reh_mp4) / 4)
                        sol.delta_reh = sol._N_reh - sol._N_end
                        sol.w_reh = np.log(3/2 * sol.V_end/sol.rho_reh_mp4) / (3*sol.delta_reh) - 1

                elif calibration_method == 'reheating':  # derive a0 and Omega_K0 from reheating
                    if Omega_K0 is not None:
                        raise ValueError(f"For curved universes with "
                                         f"calibration_method='reheating' Omega_K0 must be None, "
                                         f"but got Omega_K0={Omega_K0}.")
                    sol.N0 = (sol._N_end
                              + np.log((45/pi**2)**(1/4) * g0**(-1/3))
                              + np.log(g_th)/12
                              + np.log(sol.V_end/T_CMB_Tp**4)/4)
                    if (w_reh is None and delta_reh is None) or delta_reh == 0 or w_reh == 1/3:
                        # assume instant reheating
                        sol.delta_reh = 0
                        sol.w_reh = np.nan
                        sol.rho_reh_mp4 = 3/2 * sol.V_end
                        sol.rho_reh_GeV = (sol.rho_reh_mp4 * mp_GeV / lp_iGeV**3)**(1/4)
                        sol._N_reh = sol._N_end
                    elif w_reh is not None and delta_reh is not None:
                        if delta_reh < 0 or w_reh < -1 / 3:
                            raise ValueError(f"delta_reh must be positive (end of reheating "
                                             f"must be after end of inflation) and w_reh must "
                                             f"be greater than -1/3 (reheating by definition "
                                             f"happens after the end of inflation, but "
                                             f"w_reh<-1/3 is inflating), but got "
                                             f"delta_reh={delta_reh} and w_reh={w_reh}.")
                        sol.w_reh = w_reh
                        sol.delta_reh = delta_reh
                        sol.N0 += 3/4 * (1/3 - w_reh) * delta_reh
                        sol.rho_reh_mp4 = 3/2 * sol.V_end * np.exp(-3 * (1+w_reh) * delta_reh)
                        sol.rho_reh_GeV = (sol.rho_reh_mp4 * mp_GeV / lp_iGeV**3)**(1/4)
                        sol._N_reh = sol._N_end + delta_reh
                    elif ((w_reh is None and delta_reh is not None) or
                          (w_reh is not None and delta_reh is None)):
                        raise ValueError(f"Both w_reh and delta_reh must be given for reheating "
                                         f"(or set both to None for instant reheating), "
                                         f"but got w_reh={w_reh} and delta_reh={delta_reh}.")
                    sol.a0 = np.exp(sol.N0)
                    sol.a0_Mpc = sol.a0 * lp_m / Mpc_m
                    sol.Omega_K0 = -sol.K * c**2 / (sol.a0_Mpc * h * 100 * 1e3)**2

                else:  # only 'Omega_K0' and 'reheating' as calibration methods for K!=0
                    raise NotImplementedError(f"Unknown calibration_method={calibration_method}, "
                                              f"choose from 'Omega_K0' or 'reheating' for curved "
                                              f"universes.")

                sol.logk = sol._logaH[sol.inflation_mask] - np.log(sol.a0_Mpc)
                sol._logaH_star = np.log(K_STAR * sol.a0_Mpc)
                if np.log(K_STAR) < np.min(sol.logk) or np.log(K_STAR) > np.max(sol.logk):
                    raise InsufficientInflationError(
                        f"Pivot scale log(K_STAR)={np.log(K_STAR)} is not within the "
                        f"range of the comoving Hubble horizon during inflation: "
                        f"logk_min={np.min(sol.logk)}, logk_max={np.max(sol.logk)}."
                    )
                else:
                    logk, indices = np.unique(sol.logk, return_index=True)
                    logk2N = interp1d(logk, sol._N[sol.inflation_mask][indices])
                    sol._N_cross = logk2N(np.log(K_STAR))
                sol.N_star = sol._N_end - sol._N_cross

            # both flat and curved universes
            sol.N = sol._N + sol.delta_N_calib
            sol.N_beg = sol._N_beg + sol.delta_N_calib
            sol.N_end = sol._N_end + sol.delta_N_calib
            sol.N_reh = sol._N_reh + sol.delta_N_calib
            sol.N_cross = sol._N_cross + sol.delta_N_calib
            sol.N_dagg = sol.N_cross - sol.N_beg
            sol.k_iMpc = np.exp(sol.logk)
            sol._k = np.exp(sol._logaH[sol.inflation_mask])

            # derive comoving Hubble horizon
            sol.cHH_Mpc = sol.a0 / (np.exp(sol.N) * sol.H) * lp_m / Mpc_m
            sol.cHH_end_Mpc = sol.a0 / (np.exp(sol.N_end) * sol.H_end) * lp_m / Mpc_m

            # derive approximate primordial power spectra
            derive_approx_power()

        sol.calibrate_scale_factor = calibrate_scale_factor

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
            extrapolate = interp1d_kwargs.pop('ext', 'const')
            sol.logk2logP_s = InterpolatedUnivariateSpline(logk,
                                                           np.log(sol.P_scalar_approx[indices]),
                                                           k=spline_order, ext=extrapolate,
                                                           **interp1d_kwargs)
            sol.logk2logP_t = InterpolatedUnivariateSpline(logk,
                                                           np.log(sol.P_tensor_approx[indices]),
                                                           k=spline_order, ext=extrapolate,
                                                           **interp1d_kwargs)
            dlogPdlogk_s = sol.logk2logP_s.derivatives(np.log(K_STAR))
            dlogPdlogk_t = sol.logk2logP_t.derivatives(np.log(K_STAR))
            sol.A_s = np.exp(dlogPdlogk_s[0])
            sol.n_s = 1 + dlogPdlogk_s[1]
            sol.n_run = dlogPdlogk_s[2]
            sol.n_runrun = dlogPdlogk_s[3]
            sol.A_t = np.exp(dlogPdlogk_t[0])
            sol.n_t = dlogPdlogk_t[1]
            sol.r = sol.A_t / sol.A_s

        def P_s_approx(k):
            """Slow-roll approximation for the primordial power spectrum for scalar modes."""
            return np.exp(sol.logk2logP_s(np.log(k)))

        def P_t_approx(k):
            """Slow-roll approximation for the primordial power spectrum for tensor modes."""
            return np.exp(sol.logk2logP_t(np.log(k)))

        sol.P_s_approx = P_s_approx
        sol.P_t_approx = P_t_approx

        return sol
