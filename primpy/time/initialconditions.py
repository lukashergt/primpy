#!/usr/bin/env python
""":mod:`primpy.time.initialconditions`: initial conditions for inflation."""
import numpy as np
from scipy.optimize import root_scalar
from primpy.time.inflation import InflationEquationsT
from primpy.solver import solve
from primpy.events import InflationEvent, CollapseEvent


class InflationStartIC_NiPi(object):
    """Inflation start initial conditions given N_i, phi_i.

    Class for setting up initial conditions at the start of inflation, when
    the curvature density parameter was maximal after kinetic dominance.
    """

    def __init__(self, t_i, N_i, phi_i, K, potential, t_end=1e300):
        self.x_ini = t_i
        self.x_end = t_end
        self.N_i = N_i
        self.phi_i = phi_i
        self.equations = InflationEquationsT(K=K, potential=potential)
        self.aH_i = np.sqrt(potential.V(phi_i) / 2 * np.exp(2 * N_i) - K)
        self.Omega_Ki = -K / self.aH_i**2

    def __call__(self, y0, **ivp_kwargs):
        """Set background equations of inflation for `N`, `phi` and `dphidt` w.r.t. time `t`."""
        phi_i = self.phi_i
        V_i = self.equations.potential.V(phi_i)
        y0[self.equations.idx['N']] = self.N_i
        y0[self.equations.idx['phi']] = phi_i
        y0[self.equations.idx['dphidt']] = -np.sqrt(V_i)


class ISIC_mtN(InflationStartIC_NiPi):
    """Inflation start initial conditions given potential mass/Lambda, N_tot, and N_i."""

    def __init__(self, t_i, mass, N_tot, N_i, phi_i_bracket, K, Potential, t_end=1e300,
                 verbose=False):
        super(ISIC_mtN, self).__init__(t_i=t_i,
                                       N_i=N_i,
                                       phi_i=phi_i_bracket[-1],
                                       K=K,
                                       potential=Potential(mass),
                                       t_end=t_end)
        self.N_tot = N_tot
        self.phi_i_bracket = phi_i_bracket
        self.verbose = verbose

    def __call__(self, y0, **ivp_kwargs):
        """Set background equations of inflation w.r.t. time, optimizing for `N_tot`."""
        events = [InflationEvent(self.equations, direction=+1, terminal=False),
                  InflationEvent(self.equations, direction=-1, terminal=True),
                  CollapseEvent(self.equations)]

        def phii2Ntot(phi_i, kwargs):
            """Convert input `phi_i` to `N_tot`."""
            ic = InflationStartIC_NiPi(t_i=self.x_ini,
                                       N_i=self.N_i,
                                       phi_i=phi_i,
                                       K=self.equations.K,
                                       potential=self.equations.potential,
                                       t_end=self.x_end)
            sol = solve(ic, events=events, **kwargs)
            if np.isfinite(sol.N_tot):
                if self.verbose:
                    print("N_tot = %.15g" % sol.N_tot)
                return sol.N_tot - self.N_tot
            else:
                if np.size(sol.t_events['Collapse']) > 0:
                    return 0 - self.N_tot
                else:
                    print("sol = %s" % sol)
                    raise Exception("solve_ivp failed with message: %s" % sol.message)

        output = root_scalar(phii2Ntot, args=(ivp_kwargs,), bracket=self.phi_i_bracket)
        if self.verbose:
            print(output)
        phi_i_new = output.root
        super(ISIC_mtN, self).__init__(t_i=self.x_ini,
                                       N_i=self.N_i,
                                       phi_i=phi_i_new,
                                       K=self.equations.K,
                                       potential=self.equations.potential,
                                       t_end=self.x_end)
        super(ISIC_mtN, self).__call__(y0=y0, **ivp_kwargs)
        return phi_i_new, output


class ISIC_msNO(InflationStartIC_NiPi):
    """Inflation start initial conditions given potential mass/Lambda, N_star, N_i, Omega_K0, h."""

    def __init__(self, t_i, mass, N_star, N_i, Omega_K0, h, phi_i_bracket, K, Potential,
                 t_end=1e300, verbose=False):
        assert Omega_K0 != 0, "Curved universes only, here! Flat universes can set N_star freely."
        super(ISIC_msNO, self).__init__(t_i=t_i,
                                        N_i=N_i,
                                        phi_i=phi_i_bracket[-1],
                                        K=K,
                                        potential=Potential(mass),
                                        t_end=t_end)
        self.N_star = N_star
        self.Omega_K0 = Omega_K0
        self.h = h
        self.phi_i_bracket = phi_i_bracket
        self.verbose = verbose

    def __call__(self, y0, **ivp_kwargs):
        """Set background equations of inflation w.r.t. time, optimizing for `N_star`."""
        events = [InflationEvent(self.equations, direction=+1, terminal=False),
                  InflationEvent(self.equations, direction=-1, terminal=True),
                  CollapseEvent(self.equations)]

        def phii2Nstar(phi_i, kwargs):
            """Convert input `phi_i` to `N_star`."""
            ic = InflationStartIC_NiPi(t_i=self.x_ini,
                                       N_i=self.N_i,
                                       phi_i=phi_i,
                                       K=self.equations.K,
                                       potential=self.equations.potential,
                                       t_end=self.x_end)
            sol = solve(ic, events=events, **kwargs)
            if sol.success and np.isfinite(sol.N_tot):
                if self.verbose:
                    print("N_star = %.15g" % sol.N_star)
                sol.derive_approx_power(Omega_K0=self.Omega_K0, h=self.h)
                return sol.N_star - self.N_star
            else:
                if np.size(sol.t_events['Collapse']) > 0:
                    return 0 - self.N_star
                else:
                    print("sol = %s" % sol)
                    raise Exception("solve_ivp failed with message: %s" % sol.message)

        output = root_scalar(phii2Nstar, args=(ivp_kwargs,), bracket=self.phi_i_bracket)
        phi_i_new = output.root
        super(ISIC_msNO, self).__init__(t_i=self.x_ini,
                                        N_i=self.N_i,
                                        phi_i=phi_i_new,
                                        K=self.equations.K,
                                        potential=self.equations.potential,
                                        t_end=self.x_end)
        super(ISIC_msNO, self).__call__(y0=y0, **ivp_kwargs)
        return phi_i_new, output
