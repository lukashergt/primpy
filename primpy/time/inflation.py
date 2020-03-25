#!/usr/bin/env python
""":mod:`primpy.time.inflation`: differential equations for inflation w.r.t. time `t`."""
from warnings import warn
import numpy as np
from primpy.inflation import InflationEquations


class InflationEquationsT(InflationEquations):
    """Background equations during inflation w.r.t. time `t`.

    Solves background variables in cosmic time for curved and flat universes
    using the Klein-Gordon and Friedmann equations.

    Independent variable:
        t: cosmic time

    Dependent variables:
        N: number of e-folds
        phi: inflaton field
        dphidt: d(phi) / dt

    """

    def __init__(self, K, potential):
        super(InflationEquationsT, self).__init__(K=K, potential=potential)
        self._set_independent_variable('t')
        self.add_variable('N', 'phi', 'dphidt', 'eta')

    def __call__(self, t, y):
        """System of coupled ODE."""
        N = self.N(t, y)
        H = self.H(t, y)
        dphidt = self.dphidt(t, y)
        dVdphi = self.dVdphi(t, y)

        dy = np.zeros_like(y)
        dy[self.idx['N']] = H
        dy[self.idx['phi']] = dphidt
        dy[self.idx['dphidt']] = -3 * H * dphidt - dVdphi
        dy[self.idx['eta']] = np.exp(-N)
        return dy

    def H2(self, t, y):
        """Compute the square of the Hubble parameter using the Friedmann equation."""
        N = self.N(t, y)
        V = self.V(t, y)
        dphidt = self.dphidt(t, y)
        return (dphidt**2 / 2 + V) / 3 - self.K * np.exp(-2 * N)

    def inflating(self, t, y):
        """Inflation diagnostic for event tracking."""
        return self.V(t, y) - self.dphidt(t, y)**2

    def w(self, t, y):
        """Compute the equation of state parameter."""
        V = self.V(t, y)
        dphidt = self.dphidt(t, y)
        p = dphidt**2 / 2 - V
        rho = dphidt**2 / 2 + V
        return p / rho

    def sol(self, sol, **kwargs):
        """Post-processing of `solve_ivp` solution."""
        sol = super(InflationEquationsT, self).sol(sol, **kwargs)
        self.postprocessing_inflation_start(sol)
        self.postporcessing_inflation_end(sol)
        return sol

    def postprocessing_inflation_start(self, sol):
        """Extract starting point of inflation from event tracking."""
        # Case 1: Universe has collapsed
        if 'Collapse_dir0_term1' in sol.t_events and sol.t_events['Collapse_dir0_term1'].size > 0:
            sol.t_beg = np.nan
            sol.N_beg = np.nan
        # Case 1: inflating from the start
        elif self.inflating(sol.x[0], sol.y[:, 0]) >= 0:
            sol.t_beg = sol.t[0]
            sol.N_beg = sol.N[0]
        # Case 2: there is a transition from non-inflating to inflating
        elif 'Inflation_dir1_term0' in sol.t_events:
            sol.t_beg = sol.t_events['Inflation_dir1_term0'][0]
            sol.N_beg = sol.N[sol.t == sol.t_beg]
        else:
            warn("In order to calculate quantities such as `N_tot`, "
                 "make sure to track the event InflationEvent(ic.equations, direction=1).")

    def postporcessing_inflation_end(self, sol):
        """Extract end point of inflation from event tracking."""
        # end of inflation is first transition from inflating to non-inflating
        if 'Inflation_dir-1_term1' in sol.t_events:
            sol.t_end = sol.t_events['Inflation_dir-1_term1'][0]
        elif 'Inflation_dir-1_term0' in sol.t_events:
            sol.t_end = sol.t_events['Inflation_dir-1_term0'][0]
        # Case: inflation did not end
        elif self.inflating(sol.x[-1], sol.y[:, -1]) <= 0:
            sol.t_end = sol.t[-1]
        else:
            warn("In order to calculate quantities such as `N_tot`, "
                 "make sure to track the event InflationEvent(ic.equations, direction=-1).")
        sol.N_end = sol.N[sol.t == sol.t_end]
        sol.phi_end = sol.phi[sol.t == sol.t_end]
        sol.V_end = sol.potential.V(sol.phi_end)
