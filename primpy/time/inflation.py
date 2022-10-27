#!/usr/bin/env python
""":mod:`primpy.time.inflation`: differential equations for inflation w.r.t. time `t`."""
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
        dphidt: `d(phi)/dt`

    """

    def __init__(self, K, potential, track_eta=False, verbose=False):
        super(InflationEquationsT, self).__init__(K=K, potential=potential, verbose=verbose)
        self._set_independent_variable('t')
        self.add_variable('phi', 'dphidt', 'N')
        self.track_eta = track_eta
        if track_eta:
            self.add_variable('eta')

    def __call__(self, x, y):
        """System of coupled ODEs for underlying variables."""
        N = self.N(x, y)
        H = self.H(x, y)
        dphidt = self.dphidt(x, y)
        dVdphi = self.dVdphi(x, y)

        dy = np.zeros_like(y)
        dy[self.idx['phi']] = dphidt
        dy[self.idx['dphidt']] = -3 * H * dphidt - dVdphi
        dy[self.idx['N']] = H
        if self.track_eta:
            dy[self.idx['eta']] = np.exp(-N)
        return dy

    def H2(self, x, y):
        """Compute the square of the Hubble parameter using the Friedmann equation."""
        N = self.N(x, y)
        V = self.V(x, y)
        dphidt = self.dphidt(x, y)
        return (dphidt**2 / 2 + V) / 3 - self.K * np.exp(-2 * N)

    def w(self, x, y):
        """Compute the equation of state parameter."""
        V = self.V(x, y)
        dphidt = self.dphidt(x, y)
        p = dphidt**2 / 2 - V
        rho = dphidt**2 / 2 + V
        return p / rho

    def inflating(self, x, y):
        """Inflation diagnostic for event tracking."""
        return self.V(x, y) - self.dphidt(x, y)**2

    def sol(self, sol, **kwargs):
        """Post-processing of `solve_ivp` solution."""
        sol = super(InflationEquationsT, self).sol(sol, **kwargs)
        return sol
