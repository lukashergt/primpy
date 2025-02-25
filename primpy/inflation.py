"""General setup for equations for cosmic inflation."""
from warnings import warn
from abc import ABC
import numpy as np
from scipy.special import zeta
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

    @staticmethod
    def get_H2(N, dphi, V, K):
        """Get the Hubble parameter squared from the background equations.

        Parameters
        ----------
        N : float or array_like
            Number of e-folds `N=ln(a)`.
        dphi : float or array_like
            1st derivative of inflaton field.
        V : float or array_like
            Inflation potential at `phi`.
        K : int
            Curvature parameter.

        Returns
        -------
        H2 : float or array_like
            Hubble parameter squared.
        """

    @staticmethod
    def get_dH(N, H, dphi, K):
        """Get the 1st time derivative of the Hubble parameter from the background equations.

        Parameters
        ----------
        N : float or array_like
            Number of e-folds `N=ln(a)`.
        H : float or array_like
            Hubble parameter.
        dphi : float or array_like
            1st derivative of inflaton field.
        K : int
            Curvature parameter.

        Returns
        -------
        dH : float or array_like
            1st derivative of Hubble parameter.
        """

    def get_dH_H(N, H2, dphi, K):  # noqa: D102
        """Get the 1st time derivative of the Hubble parameter normalised by the Hubble parameter..

        Parameters
        ----------
        N : float or array_like
            Number of e-folds `N=ln(a)`.
        H2 : float or array_like
            Hubble parameter squared.
        dphi : float or array_like
            1st derivative of inflaton field.
        K : int
            Curvature parameter.

        Returns
        -------
        dH_H : float or array_like
            1st derivative of Hubble parameter normalised by Hubble parameter: `dH/H`.
        """

    @staticmethod
    def get_d2H(N, H, dH, dphi, d2phi, K):
        """Get the 2nd time derivative of the Hubble parameter from the background equations.

        Parameters
        ----------
        N : float or array_like
            Number of e-folds `N=ln(a)`.
        H : float or array_like
            Hubble parameter.
        dH : float or array_like
            1st derivative of Hubble parameter.
        dphi : float or array_like
            1st derivative of inflaton field.
        d2phi : float or array_like
            2nd derivative of inflaton field.
        K : int
            Curvature parameter.

        Returns
        -------
        d2H : float or array_like
            2nd derivative of Hubble parameter.
        """

    @staticmethod
    def get_d3H(N, H, dH, d2H, dphi, d2phi, d3phi, K):
        """Get the 3rd time derivative of the Hubble parameter from the background equations.

        Parameters
        ----------
        N : float or array_like
            Number of e-folds `N=ln(a)`.
        H : float or array_like
            Hubble parameter.
        dH : float or array_like
            1st derivative of Hubble parameter.
        d2H : float or array_like
            2nd derivative of Hubble parameter.
        dphi : float or array_like
            1st derivative of inflaton field.
        d2phi : float or array_like
            2nd derivative of inflaton field.
        d3phi : float or array_like
            3rd derivative of inflaton field.
        K : int
            Curvature parameter.

        Returns
        -------
        d3H : float or array_like
            3rd derivative of Hubble parameter.
        """

    @staticmethod
    def get_d2phi(H2, dH_H, dphi, dV):
        """Get the 2nd time derivative of the inflaton field from the background equations.

        Parameters
        ----------
        H2 : float or array_like
            Hubble parameter squared.
        dH_H : float or array_like
            1st derivative of Hubble parameter normalised by Hubble parameter `dH/H`.
        dphi : float or array_like
            1st derivative of inflaton field.
        dV : float or array_like
            1st derivative of inflation potential at `phi`.

        Returns
        -------
        d2phi : float or array_like
            2nd derivative of inflaton field.
        """

    @staticmethod
    def get_d3phi(H, dH, d2H, dphi, d2phi, dV, d2V):
        """Get the 3rd time derivative of the inflaton field from the background equations.

        Parameters
        ----------
        H : float or array_like
            Hubble parameter.
        dH : float or array_like
            1st derivative of Hubble parameter.
        d2H : float or array_like
            2nd derivative of Hubble parameter.
        dphi : float or array_like
            1st derivative of inflaton field.
        d2phi : float or array_like
            2nd derivative of inflaton field.
        dV : float or array_like
            1st derivative of inflation potential at `phi`.
        d2V : float or array_like
            2nd derivative of inflation potential at `phi`.

        Returns
        -------
        d3phi : float or array_like
            3rd derivative of inflaton field.
        """

    @staticmethod
    def get_d4phi(H, dH, d2H, d3H, dphi, d2phi, d3phi, dV, d2V, d3V):
        """Get the 4th time derivative of the inflaton field from the background equations.

        Parameters
        ----------
        H : float or array_like
            Hubble parameter.
        dH : float or array_like
            1st derivative of Hubble parameter.
        d2H : float or array_like
            2nd derivative of Hubble parameter.
        d3H : float or array_like
            3rd derivative of Hubble parameter.
        dphi : float or array_like
            1st derivative of inflaton field.
        d2phi : float or array_like
            2nd derivative of inflaton field.
        d3phi : float or array_like
            3rd derivative of inflaton field.
        dV : float or array_like
            1st derivative of inflation potential at `phi`.
        d2V : float or array_like
            2nd derivative of inflation potential at `phi`.
        d3V : float or array_like
            3rd derivative of inflation potential at `phi`.

        Returns
        -------
        d4phi : float or array_like
            4th derivative of inflaton field.
        """

    @staticmethod
    def get_epsilon_1H(H, dH):
        """Get the 1st Hubble flow parameter.

        This definition of the 1st Hubble flow parameter was suggested e.g. by
        Stewart & Lyth (1993) in eq. (23).
        https://arxiv.org/abs/gr-qc/9302019

        Parameters
        ----------
        H : float or array_like
            Hubble parameter.
        dH : float or array_like
            1st derivative of Hubble parameter.

        Returns
        -------
        epsilon_1H : float or array_like
            1st Hubble flow parameter.
        """

    @staticmethod
    def get_epsilon_2H(H, dH, d2H, kind=None):
        """Get the 2nd Hubble flow parameter.

        This definition of the 2nd Hubble flow parameter was suggested e.g. by
        Leach, Liddle, Martin & Schwarz (2003) in eq. (15).
        https://arxiv.org/abs/astro-ph/0202094

        Parameters
        ----------
        H : float or array_like
            Hubble parameter.
        dH : float or array_like
            1st derivative of Hubble parameter.
        d2H : float or array_like
            2nd derivative of Hubble parameter.
        kind : str or None, default=None
            Switch to alternative formulation given by Gong (2004).

        Returns
        -------
        epsilon_2H : float or array_like
            2nd Hubble flow parameter.
        """

    @staticmethod
    def get_epsilon_3H(H, dH, d2H, d3H, kind=None):
        """Get the 3rd Hubble flow parameter.

        This definition of the 3rd Hubble flow parameter was suggested e.g. by
        Leach, Liddle, Martin & Schwarz (2003) in eq. (15).
        https://arxiv.org/abs/astro-ph/0202094

        Parameters
        ----------
        H : float or array_like
            Hubble parameter.
        dH : float or array_like
            1st derivative of Hubble parameter.
        d2H : float or array_like
            2nd derivative of Hubble parameter.
        d3H : float or array_like
            3rd derivative of Hubble parameter.
        kind : str or None, default=None
            Switch to alternative formulation given by Gong (2004).

        Returns
        -------
        epsilon_3H : float or array_like
            3rd Hubble flow parameter.
        """

    @staticmethod
    def get_delta_1(H, dH, dphi, d2phi):
        """Get the 2nd slow-roll parameter for n=1.

        This definition of the 2nd slow roll parameter was suggested e.g. by
        Stewart & Lyth (1993) in eq. (23).
        https://arxiv.org/abs/gr-qc/9302019

        Parameters
        ----------
        H : float or array_like
            Hubble parameter.
        dH : float or array_like
            1st derivative of Hubble parameter.
        dphi : float or array_like
            1st derivative of inflaton field.
        d2phi : float or array_like
            2nd derivative of inflaton field.

        Returns
        -------
        delta_1 : float or array_like
        """

    @staticmethod
    def get_delta_2(H, dH, d2H, dphi, d2phi, d3phi):
        """Get the 2nd slow-roll parameter for n=2.

        This definition of higher orders of the 2nd slow roll parameter was suggested e.g. by
        Stewart & Gong (2001) in eq. (22).
        https://arxiv.org/abs/astro-ph/0101225

        Parameters
        ----------
        H : float or array_like
            Hubble parameter.
        dH : float or array_like
            1st derivative of Hubble parameter.
        d2H : float or array_like
            2nd derivative of Hubble parameter.
        dphi : float or array_like
            1st derivative of inflaton field.
        d2phi : float or array_like
            2nd derivative of inflaton field.
        d3phi : float or array_like
            3rd derivative of inflaton field.

        Returns
        -------
        delta_2 : float or array_like
        """

    @staticmethod
    def get_delta_3(H, dH, d2H, d3H, dphi, d2phi, d3phi, d4phi):
        """Get the 2nd slow-roll parameter for n=3.

        This definition of higher orders of the 2nd slow roll parameter was suggested e.g. by
        Stewart & Gong (2001) in eq. (22).
        https://arxiv.org/abs/astro-ph/0101225

        Parameters
        ----------
        H : float or array_like
            Hubble parameter.
        dH : float or array_like
            1st derivative of Hubble parameter.
        d2H : float or array_like
            2nd derivative of Hubble parameter.
        d3H : float or array_like
            3rd derivative of Hubble parameter.
        dphi : float or array_like
            1st derivative of inflaton field.
        d2phi : float or array_like
            2nd derivative of inflaton field.
        d3phi : float or array_like
            3rd derivative of inflaton field.
        d4phi : float or array_like
            4th derivative of inflaton field.

        Returns
        -------
        delta_3 : float or array_like
        """

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
            sol.inflation_mask = (sol._N_beg < sol._N) & (sol._N < sol._N_end)

        def calibrate_scale_factor(
                calibration_method='N_star' if self.K == 0 else 'Omega_K0',
                Omega_K0=None, h=None,                                   # for curved universes
                N_star=None, background=None,                            # for flat universes
                delta_reh=None, w_reh=None, rho_reh_GeV=None, g_th=1e2,  # for reheating
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

            background : :class:`scipy.integrate._ivp.ivp.OdeResult`, optional
                Optionally circumvent the calibration procedure by supplying a
                previously computed background solution to copy the calibration
                from, e.g. when splitting the computation into separate forward
                and backward integrations where you can provide the solution
                from the forward integration as calibrator for the backward
                integration.

            """
            if self.K == 0:  # flat universe
                if Omega_K0 is not None and Omega_K0 != 0.0:
                    raise ValueError(f"For flat universes Omega_K0 must be 0, but got "
                                     f"Omega_K0={Omega_K0}.")
                sol.a0 = 1
                sol.N0 = 0
                sol.Omega_K0 = 0

                if background is None:
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
                    if N_star is None or N_star <= 0 or N_star != background.N_star:
                        raise ValueError(f"To circumvent the calibration by providing a "
                                         f"previously computed background solution, you "
                                         f"nonetheless need to provide a matching N_star>0, but "
                                         f"got N_star={N_star} and "
                                         f"background.Nstar={background.N_star}.")
                    sol.delta_N_calib = background.delta_N_calib
                    sol._logaH_star = background._logaH_star
                    sol._N_cross = background._N_cross
                    sol._N_reh = background._N_reh
                    sol._N_end = background._N_end
                    sol.N_star = background.N_star

                sol.a0_Mpc = np.exp(sol._logaH_star) / K_STAR
                logk = sol._logaH + np.log(K_STAR) - sol._logaH_star
                _, indices = np.unique(logk, return_index=True)
                sol.inflation_mask = sol.inflation_mask & np.isin(np.arange(sol._N.size), indices)
                sol.logk = logk[sol.inflation_mask]

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

                logk = sol._logaH - np.log(sol.a0_Mpc)
                _, indices = np.unique(logk, return_index=True)
                sol.inflation_mask = sol.inflation_mask & np.isin(np.arange(sol._N.size), indices)
                sol.logk = logk[sol.inflation_mask]
                sol._logaH_star = np.log(K_STAR * sol.a0_Mpc)
                if np.log(K_STAR) < np.min(sol.logk) or np.log(K_STAR) > np.max(sol.logk):
                    raise InsufficientInflationError(
                        f"Pivot scale log(K_STAR)={np.log(K_STAR)} is not within the "
                        f"range of the comoving Hubble horizon during inflation: "
                        f"logk_min={np.min(sol.logk)}, logk_max={np.max(sol.logk)}."
                    )
                else:
                    logk2N = interp1d(sol.logk, sol._N[sol.inflation_mask])
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
            if background is None:  # only derive if not copied from background
                derive_approx_power()

        sol.calibrate_scale_factor = calibrate_scale_factor

        def derive_approx_power(method='CGS', order=3, **interp1d_kwargs):
            """Derive the approximate primordial power spectra for scalar and tensor modes."""
            if method == 'CGS':
                derive_approx_power_CGS(order=order, **interp1d_kwargs)
            elif method == 'LLMS':
                derive_approx_power_LLMS(**interp1d_kwargs)
            elif method == 'STE':
                derive_approx_power_STE(**interp1d_kwargs)
            elif method == 'ARBDS':
                derive_approx_power_ARBDS(order=order, **interp1d_kwargs)

            dlogPdlogk_s = sol.logk2logP_s.derivatives(np.log(K_STAR))
            dlogPdlogk_t = sol.logk2logP_t.derivatives(np.log(K_STAR))
            sol.A_s = np.exp(dlogPdlogk_s[0])
            sol.n_s = 1 + dlogPdlogk_s[1]
            sol.n_run = dlogPdlogk_s[2]
            sol.n_runrun = dlogPdlogk_s[3]
            sol.A_t = np.exp(dlogPdlogk_t[0])
            sol.n_t = dlogPdlogk_t[1]
            sol.r = sol.A_t / sol.A_s

        def derive_approx_power_CGS(order=3, **interp1d_kwargs):
            """Slow-roll approximation by Choe, Gong, and Stewart.

            Relevant papers
            ---------------
            * Stewart & Gong (2001)
              http://arxiv.org/abs/astro-ph/0101225

            * Choe, Gong & Stewart (2004)
              https://arxiv.org/abs/hep-ph/0405155

            * Gong (2004)
              https://arxiv.org/abs/gr-qc/0408039

            """
            spline_order = interp1d_kwargs.pop('k', 3)
            extrapolate = interp1d_kwargs.pop('ext', 'const')

            K = sol.K
            _N = sol._N[sol.inflation_mask]
            H = sol.H[sol.inflation_mask]
            phi = sol.phi[sol.inflation_mask]
            dV = sol.potential.dV(phi)
            d2V = sol.potential.d2V(phi)
            d3V = sol.potential.d3V(phi)
            if hasattr(sol, 'dphidt'):
                dphi = sol.dphidt[sol.inflation_mask]
            else:
                dphi = sol.dphidN[sol.inflation_mask]
            dH = self.get_dH(N=_N, H=H, dphi=dphi, K=K)
            dH_H = self.get_dH_H(N=_N, H2=H**2, dphi=dphi, K=K)
            d2phi = self.get_d2phi(H2=H**2, dH_H=dH_H, dphi=dphi, dV=dV)
            d2H = self.get_d2H(N=_N, H=H, dH=dH, dphi=dphi, d2phi=d2phi, K=K)
            d3phi = self.get_d3phi(H=H, dH=dH, d2H=d2H, dphi=dphi, d2phi=d2phi, dV=dV, d2V=d2V)
            d3H = self.get_d3H(N=_N, H=H, dH=dH, d2H=d2H, dphi=dphi, d2phi=d2phi, d3phi=d3phi, K=K)
            d4phi = self.get_d4phi(H=H, dH=dH, d2H=d2H, d3H=d3H,
                                   dphi=dphi, d2phi=d2phi, d3phi=d3phi,
                                   dV=dV, d2V=d2V, d3V=d3V)

            # Stewart and Gong (2001), eq. (6) and (22) and
            # Choe, Gong & Stewart (2004), eq. (57)
            e1 = self.get_epsilon_1H(H=H, dH=dH)
            d1 = self.get_delta_1(H=H, dH=dH, dphi=dphi, d2phi=d2phi)
            d2 = self.get_delta_2(H=H, dH=dH, d2H=d2H, dphi=dphi, d2phi=d2phi, d3phi=d3phi)
            d3 = self.get_delta_3(H=H, dH=dH, d2H=d2H, d3H=d3H,
                                  dphi=dphi, d2phi=d2phi, d3phi=d3phi, d4phi=d4phi)

            # Stewart and Gong (2001), eq. (41)
            # Choe, Gong & Stewart (2004), eq. (60)
            Ps0 = H**2 / (8 * pi**2 * e1)
            alpha = 2 - np.log(2) - np.euler_gamma
            astar = alpha - e1  # from Choe, Gong & Stewart (2004) just after eq. (60)
            order1_s = (1 + (4 * astar - 2) * e1 + 2 * astar * d1 + (-astar**2 + pi**2 / 12) * d2
                        + (1 / 3 * astar**3 - pi**2 / 12 * astar + 4 / 3 - 2 / 3 * zeta(3)) * d3)
            # (slight differences between SG and CGS in 2nd order numeric coefficients)
            order2_s = (
                    # + (4 * alpha**2 - 23 + 7 * pi**2 / 3) * e1**2
                    + (4 * astar**2 - 19 + 7 * pi**2 / 3) * e1**2
                    # + (3 * alpha**2 + 2 * alpha - 22 + 29 * pi**2 / 12) * e1 * d1
                    + (3 * astar**2 + 2 * astar - 20 + 29 * pi**2 / 12) * e1 * d1
                    + (3 * astar**2 - 4 + 5 * pi**2 / 12) * d1**2
                    + (-5/3 * astar**3 - 2 * astar**2 + 20 * astar - 9/4 * pi**2 * astar
                       + 16/3 + pi**2 / 6 - 14/3 * zeta(3)) * e1 * d2
                    + (-3*astar**3 + 8*astar - 7/12 * pi**2 * astar - 4 + 2 * zeta(3)) * d1 * d2
            )
            order3_s = (
                + (4 * astar**2 + 8 * astar + 16 + 5 * pi**2 - 48 * zeta(3)) * e1**3
                + (-5 / 3 * astar**3 + 4 * astar**2 + 32 * astar - 9 / 4 * pi**2 * astar
                   + 88 / 3 + 23 / 3 * pi**2 - 230 / 3 * zeta(3)) * e1**2 * d1
                + (3 * astar**3 + 4 * astar**2 - 24 * astar + 13 / 4 * pi**2 * astar
                   + 16 + 7 / 3 * pi**2 - 30 * zeta(3)) * e1 * d1**2
                + (4 * astar**3 - 16 * astar + 5 / 3 * pi**2 * astar + 8 - 6 * zeta(3)) * d1**3
            )

            # Gong (2004), eq. (23)
            e2 = self.get_epsilon_2H(H=H, dH=dH, d2H=d2H, kind='Gong')
            e3 = self.get_epsilon_3H(H=H, dH=dH, d2H=d2H, d3H=d3H, kind='Gong')

            # Gong (2004), eq. (25)
            Pt0 = 2 * (H / pi)**2
            order1_t = (
                1 + 2 * (astar - 1) * e1
                + (-astar**2 + 2 * astar - 2 + pi**2/12) * e2
                + (astar**3/3 - astar**2 + (2-pi**2/12)*astar + pi**2/12 - 2/3 - 2/3*zeta(3)) * e3
            )
            order2_t = (
                + (2 * astar**2 - 2 * astar - 3 + pi**2/2) * e1**2
                + (-5/3 * astar**3 + 2 * astar**2 + (8 - 11/12 * pi**2) * astar + 7/6 * pi**2
                   - 26/3 - 2/3 * zeta(3)) * e1 * e2
            )
            order3_t = (4/3 * astar**3 - 8 * astar + pi**2 * astar + 16/3 - 14/3 * zeta(3)) * e1**3

            # order 0
            mask = (Ps0 > 0) & (Pt0 > 0)
            logP_s = np.log(Ps0[mask])
            logP_t = np.log(Pt0[mask])
            if order == 0:
                sol.P_scalar_approx = Ps0
                sol.P_tensor_approx = Pt0
            logk2logP_s_0 = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_s,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            logk2logP_t_0 = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_t,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            sol.P_s_approx_CGS0 = lambda k: np.exp(logk2logP_s_0(np.log(k)))
            sol.P_t_approx_CGS0 = lambda k: np.exp(logk2logP_t_0(np.log(k)))

            # order 1
            P_s = Ps0 * order1_s
            P_t = Pt0 * order1_t
            mask = (P_s > 0) & (P_t > 0)
            logP_s = np.log(P_s[mask])
            logP_t = np.log(P_t[mask])
            if order == 1:
                sol.P_scalar_approx = P_s
                sol.P_tensor_approx = P_t
            logk2logP_s_1 = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_s,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            logk2logP_t_1 = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_t,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            sol.P_s_approx_CGS1 = lambda k: np.exp(logk2logP_s_1(np.log(k)))
            sol.P_t_approx_CGS1 = lambda k: np.exp(logk2logP_t_1(np.log(k)))

            # order 2
            P_s = Ps0 * (order1_s + order2_s)
            P_t = Pt0 * (order1_t + order2_t)
            mask = (P_s > 0) & (P_t > 0)
            logP_s = np.log(P_s[mask])
            logP_t = np.log(P_t[mask])
            if order == 2:
                sol.P_scalar_approx = P_s
                sol.P_tensor_approx = P_t
            logk2logP_s_2 = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_s,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            logk2logP_t_2 = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_t,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            sol.P_s_approx_CGS2 = lambda k: np.exp(logk2logP_s_2(np.log(k)))
            sol.P_t_approx_CGS2 = lambda k: np.exp(logk2logP_t_2(np.log(k)))

            # order 3
            P_s = Ps0 * (order1_s + order2_s + order3_s)
            P_t = Pt0 * (order1_t + order2_t + order3_t)
            mask = (P_s > 0) & (P_t > 0)
            logP_s = np.log(P_s[mask])
            logP_t = np.log(P_t[mask])
            if order == 3:
                sol.P_scalar_approx = P_s
                sol.P_tensor_approx = P_t
            logk2logP_s_3 = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_s,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            logk2logP_t_3 = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_t,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            sol.P_s_approx_CGS3 = lambda k: np.exp(logk2logP_s_3(np.log(k)))
            sol.P_t_approx_CGS3 = lambda k: np.exp(logk2logP_t_3(np.log(k)))
            if order == 0:
                sol.logk2logP_s = logk2logP_s_0
                sol.logk2logP_t = logk2logP_t_0
            elif order == 1:
                sol.logk2logP_s = logk2logP_s_1
                sol.logk2logP_t = logk2logP_t_1
            elif order == 2:
                sol.logk2logP_s = logk2logP_s_2
                sol.logk2logP_t = logk2logP_t_2
            elif order == 3:
                sol.logk2logP_s = logk2logP_s_3
                sol.logk2logP_t = logk2logP_t_3

        def derive_approx_power_LLMS(**interp1d_kwargs):
            """Slow-roll approximation by Leach, Liddle, Martin, and Schwarz (2002).

            http://arxiv.org/abs/astro-ph/0101225v2

            """
            spline_order = interp1d_kwargs.pop('k', 3)
            extrapolate = interp1d_kwargs.pop('ext', 'const')

            K = sol.K
            _N = sol._N[sol.inflation_mask]
            H = sol.H[sol.inflation_mask]
            phi = sol.phi[sol.inflation_mask]
            dV = sol.potential.dV(phi)
            d2V = sol.potential.d2V(phi)
            if hasattr(sol, 'dphidt'):
                dphi = sol.dphidt[sol.inflation_mask]
            else:
                dphi = sol.dphidN[sol.inflation_mask]
            dH = self.get_dH(N=_N, H=H, dphi=dphi, K=K)
            dH_H = self.get_dH_H(N=_N, H2=H**2, dphi=dphi, K=K)
            d2phi = self.get_d2phi(H2=H**2, dH_H=dH_H, dphi=dphi, dV=dV)
            d2H = self.get_d2H(N=_N, H=H, dH=dH, dphi=dphi, d2phi=d2phi, K=K)
            d3phi = self.get_d3phi(H=H, dH=dH, d2H=d2H, dphi=dphi, d2phi=d2phi, dV=dV, d2V=d2V)
            d3H = self.get_d3H(N=_N, H=H, dH=dH, d2H=d2H, dphi=dphi, d2phi=d2phi, d3phi=d3phi, K=K)

            # Leach, Liddle, Martin, and Schwarz (2002), eq. (15)
            e1 = self.get_epsilon_1H(H=H, dH=dH)
            e2 = self.get_epsilon_2H(H=H, dH=dH, d2H=d2H)
            e3 = self.get_epsilon_3H(H=H, dH=dH, d2H=d2H, d3H=d3H)
            N2H = interp1d(_N, H)
            H_star = N2H(sol._N_cross)
            N2e1 = interp1d(_N, e1)
            e1 = N2e1(sol._N_cross)
            N2e2 = interp1d(_N, e2)
            e2 = N2e2(sol._N_cross)
            N2e3 = interp1d(_N, e3)
            e3 = N2e3(sol._N_cross)

            # Leach, Liddle, Martin, and Schwarz (2002), eqs. (24), (25)
            # Note that they use m_Pl**2=8pi, while I use m_Pl=1
            Ps0 = H_star**2 / (8 * pi**2 * e1)
            Pt0 = 2 * (H_star / pi)**2

            # Leach, Liddle, Martin, and Schwarz (2002), eqs. (15), (34)-(41)
            C = np.euler_gamma + np.log(2) - 2
            bs0 = (-2 * (C + 1) * e1 - C * e2
                   + (-2 * C + pi**2 / 2 - 7) * e1**2
                   + (-C**2 - 3 * C + 7 * pi**2 / 12 - 7) * e1 * e2
                   + (pi**2 / 8 - 1) * e2**2
                   + (-C**2 / 2 + pi**2 / 24) * e2 * e3)
            n_s = 1 - 2 * e1 - e2 - 2 * e1**2 - (2 * C + 3) * e1 * e2 - C * e2 * e3
            n_s_run = -2 * e1 * e2 - e2 * e3
            bt0 = (-2 * (C + 1) * e1
                   + (-2 * C + pi**2 / 2 - 7) * e1**2
                   + (-C**2 - 2 * C + pi**2 / 12 - 2) * e1 * e2)
            n_t = -2 * e1 - 2 * e1**2 - 2 * (C + 1) * e1 * e2
            n_t_run = -2 * e1 * e2

            log_k_kstar = sol.logk - np.log(K_STAR)
            logP_s = np.log(Ps0) + bs0 + (n_s - 1) * log_k_kstar + n_s_run / 2 * log_k_kstar**2
            logP_t = np.log(Pt0) + bt0 + n_t * log_k_kstar + n_t_run / 2 * log_k_kstar**2
            logk2logP_s_LLMS = InterpolatedUnivariateSpline(
                sol.logk, logP_s,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            logk2logP_t_LLMS = InterpolatedUnivariateSpline(
                sol.logk, logP_t,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            sol.P_s_approx_LLMS = lambda k: np.exp(logk2logP_s_LLMS(np.log(k)))
            sol.P_t_approx_LLMS = lambda k: np.exp(logk2logP_t_LLMS(np.log(k)))
            sol.P_scalar_approx = np.exp(logP_s)
            sol.P_tensor_approx = np.exp(logP_t)
            sol.logk2logP_s = logk2logP_s_LLMS
            sol.logk2logP_t = logk2logP_t_LLMS

        def derive_approx_power_STE(**interp1d_kwargs):
            """Slow-roll approximation by Schwarz and Terrero-Escalante (2004).

            https://arxiv.org/abs/hep-ph/0403129

            """
            spline_order = interp1d_kwargs.pop('k', 3)
            extrapolate = interp1d_kwargs.pop('ext', 'const')

            K = sol.K
            _N = sol._N[sol.inflation_mask]
            H = sol.H[sol.inflation_mask]
            phi = sol.phi[sol.inflation_mask]
            dV = sol.potential.dV(phi)
            d2V = sol.potential.d2V(phi)
            if hasattr(sol, 'dphidt'):
                dphi = sol.dphidt[sol.inflation_mask]
            else:
                dphi = sol.dphidN[sol.inflation_mask]
            dH = self.get_dH(N=_N, H=H, dphi=dphi, K=K)
            dH_H = self.get_dH_H(N=_N, H2=H**2, dphi=dphi, K=K)
            d2phi = self.get_d2phi(H2=H**2, dH_H=dH_H, dphi=dphi, dV=dV)
            d2H = self.get_d2H(N=_N, H=H, dH=dH, dphi=dphi, d2phi=d2phi, K=K)
            d3phi = self.get_d3phi(H=H, dH=dH, d2H=d2H, dphi=dphi, d2phi=d2phi, dV=dV, d2V=d2V)
            d3H = self.get_d3H(N=_N, H=H, dH=dH, d2H=d2H, dphi=dphi, d2phi=d2phi, d3phi=d3phi, K=K)

            # Schwarz and Terrero-Escalante (2004), eq. (3)
            e1 = self.get_epsilon_1H(H=H, dH=dH)
            e2 = self.get_epsilon_2H(H=H, dH=dH, d2H=d2H)
            e3 = self.get_epsilon_3H(H=H, dH=dH, d2H=d2H, d3H=d3H)

            # Schwarz and Terrero-Escalante (2004), eqs. (13) and (14)
            Ps0 = H**2 / (8 * pi**2 * e1)
            Pt0 = 2 * (H / pi)**2

            # Schwarz and Terrero-Escalante (2004), eqs. (19) and (20)
            C = np.euler_gamma + np.log(2) - 2
            as0 = (
                1 - 2 * (C + 1) * e1 - C * e2
                + (2 * C**2 + 2 * C + pi**2 / 2 - 5) * e1**2
                + (C**2 / 2 + pi**2 / 8 - 1) * e2**2
                + (C**2 - C + 7/12 * pi**2 - 7) * e1 * e2
                + (-C**2 / 2 + pi**2 / 24) * e2 * e3
            )
            at0 = (
                1 - 2 * (C + 1) * e1
                + (2 * C**2 + 2 * C + pi**2 / 2 - 5) * e1**2
                + (-C**2 - 2 * C + pi**2 / 12 - 2) * e1 * e2
            )

            P_s = Ps0 * as0
            P_t = Pt0 * at0
            mask = (P_s > 0) & (P_t > 0)
            logP_s = np.log(P_s[mask])
            logP_t = np.log(P_t[mask])
            logk2logP_s_STE = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_s,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            logk2logP_t_STE = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_t,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            sol.P_s_approx_STE = lambda k: np.exp(logk2logP_s_STE(np.log(k)))
            sol.P_t_approx_STE = lambda k: np.exp(logk2logP_t_STE(np.log(k)))
            sol.P_scalar_approx = P_s
            sol.P_tensor_approx = P_t
            sol.logk2logP_s = logk2logP_s_STE
            sol.logk2logP_t = logk2logP_t_STE

        def derive_approx_power_ARBDS(order=3, **interp1d_kwargs):
            """Slow-roll approximation up to third order (N3LO).

            Slow-roll approximation by
            Auclair & Ringeval (2022)
            https://arxiv.org/abs/2205.12608

            and

            Ballardini, Davoli, and Sirletti (2025).
            https://arxiv.org/abs/2408.05210

            """
            spline_order = interp1d_kwargs.pop('k', 3)
            extrapolate = interp1d_kwargs.pop('ext', 'const')

            K = sol.K
            _N = sol._N[sol.inflation_mask]
            H = sol.H[sol.inflation_mask]
            phi = sol.phi[sol.inflation_mask]
            dV = sol.potential.dV(phi)
            d2V = sol.potential.d2V(phi)
            if hasattr(sol, 'dphidt'):
                dphi = sol.dphidt[sol.inflation_mask]
            else:
                dphi = sol.dphidN[sol.inflation_mask]
            dH = self.get_dH(N=_N, H=H, dphi=dphi, K=K)
            dH_H = self.get_dH_H(N=_N, H2=H**2, dphi=dphi, K=K)
            d2phi = self.get_d2phi(H2=H**2, dH_H=dH_H, dphi=dphi, dV=dV)
            d2H = self.get_d2H(N=_N, H=H, dH=dH, dphi=dphi, d2phi=d2phi, K=K)
            d3phi = self.get_d3phi(H=H, dH=dH, d2H=d2H, dphi=dphi, d2phi=d2phi, dV=dV, d2V=d2V)
            d3H = self.get_d3H(N=_N, H=H, dH=dH, d2H=d2H, dphi=dphi, d2phi=d2phi, d3phi=d3phi, K=K)

            # Auclair & Ringeval (2022), eqs. (1)
            # Ballardini, Davoli, and Sirletti (2025), eqs. (2)
            e1 = self.get_epsilon_1H(H=H, dH=dH)
            e2 = self.get_epsilon_2H(H=H, dH=dH, d2H=d2H)
            e3 = self.get_epsilon_3H(H=H, dH=dH, d2H=d2H, d3H=d3H)
            e4 = 0  # TODO
            N2H = interp1d(_N, H)
            H_star = N2H(sol._N_cross)
            N2e1 = interp1d(_N, e1)
            e1 = N2e1(sol._N_cross)
            N2e2 = interp1d(_N, e2)
            e2 = N2e2(sol._N_cross)
            N2e3 = interp1d(_N, e3)
            e3 = N2e3(sol._N_cross)
            alpha = 2 - np.log(2) - np.euler_gamma
            Z = zeta(3) / 3 * 7  # factor 7 matches Auclair & Ringeval (2022) and matches LLMS

            # Auclair & Ringeval (2022), eq. (54)
            # Ballardini, Davoli, and Sirletti (2025), eq. (45) or eq. (C.2)
            Ps0 = H_star**2 / (8 * pi**2 * e1)
            Pt0 = 2 * (H_star / pi)**2

            # Auclair & Ringeval (2022), eq. (54)
            # Ballardini, Davoli, and Sirletti (2025), eq. (45) or eqs. (C.3) to (C.6)
            as0_1 = (
                1 - 2 * (1-alpha) * e1 + alpha * e2
            )
            as0_2 = (
                + (-3 - 2 * alpha + 2 * alpha**2 + pi**2/2) * e1**2
                + (-6 + alpha + alpha**2 + 7 * pi**2 / 12) * e1 * e2
                + (-8 + 4 * alpha**2 + pi**2) / 8 * e2**2
                + (-12 * alpha**2 + pi**2) / 24 * e2 * e3
            )
            as0_3 = (
                - (-16 + 24 * alpha - 4*alpha**3 - 3*alpha*pi**2 + 6*Z) / 24 * (8 * e1**3 + e2**3)
                + (-72*alpha + 36*alpha**2 + 13*pi**2 + 8*alpha*pi**2 - 36*Z) / 12 * e1**2 * e2
                - (16+24*alpha-12*alpha**2-8*alpha**3-15*pi**2-6*alpha*pi**2+84*Z) / 24 * e1*e2**2
                + (16 + 4*alpha**3 - alpha*pi**2 - 24*Z) / 24 * (e2*e3**2 + e2 * e3 * e4)
                + (48*alpha - 12*alpha**3 - 5*alpha*pi**2) / 24 * e2**2 * e3
                + (-8+72*alpha-12*alpha**2-8*alpha**3+pi**2-6*alpha*pi**2-24*Z) / 12 * e1 * e2 * e3
            )
            as1_1 = (
                -2 * e1 - e2
            )
            as1_2 = (
                + 2 * (-2 * alpha + 1) * e1**2
                + (-2 * alpha - 1) * e1 * e2
                - alpha * e2**2
                + alpha * e2 * e3
            )
            as1_3 = (
                - (-8 + 4*alpha**2 + pi**2) / 8 * (8 * e1**3 + e2**3)
                - 2/3 * (-9 + 9*alpha + pi**2) * e1**2 * e2
                - (-4 + 4*alpha + 4*alpha**2 + pi**2) / 4 * e1 * e2**2
                + (-12 + 4*alpha + 4*alpha**2 + pi**2) / 2 * e1 * e2 * e3
                + (-12*alpha**2 + pi**2) / 24 * (e2 * e3**2 + e2 * e3 * e4)
                + (-48 + 36*alpha**2 + 5*pi**2) / 24 * e2**2 * e3
            )
            as2_2 = 1 / 2 * (
                4 * e1**2 + 2 * e1 * e2 + e2**2 - e2 * e3
            )
            as2_3 = 1 / 2 * (
                + 6 * e1**2 * e2 + (1 + 2*alpha) * (e1 * e2**2 - 2 * e1 * e2 * e3)
                + alpha * (8 * e1**3 + e2**3 - 3 * e2**2 * e3 + e2 * e3**2 + e2 * e3 * e4)
            )
            as3_3 = 1 / 6 * (
                    - 8 * e1**3 - 2 * e1 * e2**2 - e2**3 + 4 * e1 * e2 * e3
                    + 3 * e2**2 * e3 - e2 * e3**2 - e2 * e3 * e4
            )

            # Auclair & Ringeval (2022), eq. (44)
            # Ballardini, Davoli, and Sirletti (2025), eq. (55) or eqs. (C.7) to (C.10)
            at0_1 = 1 + 2 * (-1 + alpha) * e1
            at0_2 = (
                + (-3 - 2 * alpha + 2 * alpha**2 + pi**2 / 2) * e1**2
                + (-2 + 2 * alpha - alpha**2 + pi**2 / 12) * e1 * e2
            )
            at0_3 = (
                # -1/3 * (-16 + 24*alpha - 4*alpha**3 - 3*alpha*pi**2 + 6*Z) * e1**3
                - 1/3 * (-16 + 24*alpha - 4*alpha**3 - 3*alpha*pi**2 + 6*Z) * e1**3
                + 1/12 * (-96+72*alpha+36*alpha**2-24*alpha**3+13*pi**2-10*alpha*pi**2) * e1**2*e2
                - 1/12 * (8 - 24 * alpha + 12 * alpha**2 - 4 * alpha**3 - pi**2
                          + alpha * pi**2 + 24 * Z) * (e1 * e2**2 + e1 * e2 * e3)
            )
            at1_1 = -2 * e1
            at1_2 = 2 * (-2 * alpha + 1) * e1**2 + (-2 + 2 * alpha) * e1 * e2
            at1_3 = (
                - (-8 + 4 * alpha**2 + pi**2) * e1**3
                + 6 * (-1 - alpha + alpha**2 + 5 * pi**2 / 36) * e1**2 * e2
                + (-2 + 2 * alpha - alpha**2 + pi**2 / 12) * (e1 * e2**2 + e1 * e2 * e3)
            )
            at2_2 = 1/2 * (4 * e1**2 - 2 * e1 * e2)
            at2_3 = 1/2 * (
                + 8 * alpha * e1**3
                + 2 * (3 - 6 * alpha) * e1**2 * e2
                - 2 * (1 - alpha) * (e1 * e2**2 + e1 * e2 * e3)
            )
            at3_3 = 1/6 * (-8 * e1**3 + 12 * e1**2 * e2 - 2 * e1 * e2**2 - 2 * e1 * e2 * e3)

            # order 1
            log_k_kstar = sol.logk - np.log(K_STAR)
            P_s = Ps0 * (as0_1 + as1_1 * log_k_kstar)
            P_t = Pt0 * (at0_1 + at1_1 * log_k_kstar)
            mask = (P_s > 0) & (P_t > 0)
            logP_s = np.log(P_s[mask])
            logP_t = np.log(P_t[mask])
            logk2logP_s_ARBDS1 = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_s,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            logk2logP_t_ARBDS1 = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_t,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            if order == 1:
                sol.P_scalar_approx = P_s
                sol.P_tensor_approx = P_t
            sol.P_s_approx_ARBDS1 = lambda k: np.exp(logk2logP_s_ARBDS1(np.log(k)))
            sol.P_t_approx_ARBDS1 = lambda k: np.exp(logk2logP_t_ARBDS1(np.log(k)))

            # order 2
            P_s = Ps0 * (as0_1 + as0_2 + (as1_1 + as1_2) * log_k_kstar + as2_2 * log_k_kstar**2)
            P_t = Pt0 * (at0_1 + at0_2 + (at1_1 + at1_2) * log_k_kstar + at2_2 * log_k_kstar**2)
            mask = (P_s > 0) & (P_t > 0)
            logP_s = np.log(P_s[mask])
            logP_t = np.log(P_t[mask])
            logk2logP_s_ARBDS2 = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_s,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            logk2logP_t_ARBDS2 = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_t,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            if order == 2:
                sol.P_scalar_approx = P_s
                sol.P_tensor_approx = P_t
            sol.P_s_approx_ARBDS2 = lambda k: np.exp(logk2logP_s_ARBDS2(np.log(k)))
            sol.P_t_approx_ARBDS2 = lambda k: np.exp(logk2logP_t_ARBDS2(np.log(k)))

            # order 3
            as0 = as0_1 + as0_2 + as0_3
            at0 = at0_1 + at0_2 + at0_3
            as1 = as1_1 + as1_2 + as1_3
            at1 = at1_1 + at1_2 + at1_3
            as2 = as2_2 + as2_3
            at2 = at2_2 + at2_3
            as3 = as3_3
            at3 = at3_3
            P_s = Ps0 * (as0 + as1 * log_k_kstar + as2 * log_k_kstar**2 + as3 * log_k_kstar**3)
            P_t = Pt0 * (at0 + at1 * log_k_kstar + at2 * log_k_kstar**2 + at3 * log_k_kstar**3)
            mask = (P_s > 0) & (P_t > 0)
            logP_s = np.log(P_s[mask])
            logP_t = np.log(P_t[mask])
            logk2logP_s_ARBDS3 = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_s,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            logk2logP_t_ARBDS3 = InterpolatedUnivariateSpline(
                sol.logk[mask], logP_t,
                k=spline_order, ext=extrapolate, **interp1d_kwargs
            )
            if order == 3:
                sol.P_scalar_approx = P_s
                sol.P_tensor_approx = P_t
            sol.P_s_approx_ARBDS3 = lambda k: np.exp(logk2logP_s_ARBDS3(np.log(k)))
            sol.P_t_approx_ARBDS3 = lambda k: np.exp(logk2logP_t_ARBDS3(np.log(k)))
            if order == 1:
                sol.logk2logP_s = logk2logP_s_ARBDS1
                sol.logk2logP_t = logk2logP_t_ARBDS1
            elif order == 2:
                sol.logk2logP_s = logk2logP_s_ARBDS2
                sol.logk2logP_t = logk2logP_t_ARBDS2
            elif order == 3:
                sol.logk2logP_s = logk2logP_s_ARBDS3
                sol.logk2logP_t = logk2logP_t_ARBDS3

        sol.derive_approx_power = derive_approx_power

        def P_s_approx(k, method='CGS', order=3, **interp_kwargs):
            """Slow-roll approximation for the primordial power spectrum for scalar modes.

            Parameters
            ----------
            k : array_like
                Wavenumber in Mpc^-1.

            method : str, default='CGS'
                Choice of approximation:
                * `CGS`: Choe, Gong & Stewart (2004)
                * `LLMS`: Leach, Liddle, Martin & Schwarz (2003)
                * `STE`: Schwarz and Terrero-Escalante (2004)
                * `ARBDS`: Auclair & Ringeval (2022) and Ballardini, Davoli & Sirletti (2025)

            order : int, default=3
                The `CGS` and `ARBDS` methods are implemented in different orders or approximation.
                Ignored for other methods.

            Returns
            -------
            P_s : array_like
                Primordial power spectrum of scalar modes.

            """
            if method == 'CGS':
                if order == 3:
                    return sol.P_s_approx_CGS3(k)
                elif order == 0:
                    return sol.P_s_approx_CGS0(k)
                elif order == 1:
                    return sol.P_s_approx_CGS1(k)
                elif order == 2:
                    return sol.P_s_approx_CGS2(k)
            elif method == 'LLMS':
                if not hasattr(sol, "P_s_approx_LLMS"):
                    derive_approx_power_LLMS(**interp_kwargs)
                return sol.P_s_approx_LLMS(k)
            elif method == 'STE':
                if not hasattr(sol, "P_s_approx_STE"):
                    derive_approx_power_STE(**interp_kwargs)
                return sol.P_s_approx_STE(k)
            elif method == 'ARBDS':
                if not hasattr(sol, "P_s_approx_ARBDS"):
                    derive_approx_power_ARBDS(order=order, **interp_kwargs)
                if order == 3:
                    return sol.P_s_approx_ARBDS3(k)
                elif order == 1:
                    return sol.P_s_approx_ARBDS1(k)
                elif order == 2:
                    return sol.P_s_approx_ARBDS2(k)

            return np.exp(sol.logk2logP_s(np.log(k)))

        def P_t_approx(k, method='CGS', order=3, **interp_kwargs):
            """Slow-roll approximation for the primordial power spectrum for tensor modes.

            Parameters
            ----------
            k : array_like
                Wavenumber in Mpc^-1.

            method : str, default='CGS'
                Choice of approximation:
                * `CGS`: Gong (2004)
                * `LLMS`: Leach, Liddle, Martin & Schwarz (2003)
                * `STE`: Schwarz and Terrero-Escalante (2004)
                * `ARBDS`: Auclair & Ringeval (2022) and Ballardini, Davoli & Sirletti (2025)

            order : int, default=3
                The `CGS` and `ARBDS` methods are implemented in different orders or approximation.
                Ignored for other methods.

            Returns
            -------
            P_t : array_like
                Primordial power spectrum of tensor modes.

            """
            if method == 'CGS':
                if order is None or order == 3:
                    return sol.P_t_approx_CGS3(k)
                elif order == 0:
                    return sol.P_t_approx_CGS0(k)
                elif order == 1:
                    return sol.P_t_approx_CGS1(k)
                elif order == 2:
                    return sol.P_t_approx_CGS2(k)
            elif method == 'LLMS':
                if not hasattr(sol, "P_t_approx_LLMS"):
                    derive_approx_power_LLMS(**interp_kwargs)
                return sol.P_t_approx_LLMS(k)
            elif method == 'STE':
                if not hasattr(sol, "P_t_approx_STE"):
                    derive_approx_power_STE(**interp_kwargs)
                return sol.P_t_approx_STE(k)
            elif method == 'ARBDS':
                if not hasattr(sol, "P_t_approx_ARBDS"):
                    derive_approx_power_ARBDS(order=order, **interp_kwargs)
                if order == 3:
                    return sol.P_t_approx_ARBDS3(k)
                elif order == 1:
                    return sol.P_t_approx_ARBDS1(k)
                elif order == 2:
                    return sol.P_t_approx_ARBDS2(k)

            return np.exp(sol.logk2logP_t(np.log(k)))

        sol.P_s_approx = P_s_approx
        sol.P_t_approx = P_t_approx

        return sol
