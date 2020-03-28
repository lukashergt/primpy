#!/usr/bin/env python
""":mod:`primpy.time.inflation`: differential equations for inflation w.r.t. time `t`."""
from warnings import warn
import numpy as np
from primpy.equations import Equations
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
        return sol
