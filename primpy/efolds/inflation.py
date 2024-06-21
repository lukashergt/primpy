"""Differential equations for inflation w.r.t. e-folds `N`."""
import numpy as np
from primpy.inflation import InflationEquations


class InflationEquationsN(InflationEquations):
    """Background equations during inflation w.r.t. e-folds `N`.

    Solves background variables with e-folds `N` of the scale factor as
    independent variable for curved and flat universes using the Klein-Gordon
    and Friedmann equations.

    Independent variable:
        ``_N``: e-folds of the scale-factor
        (the underscore here means that this is the as of yet uncalibrated scale factor)

    Dependent variables:
        ``phi``: inflaton field
        ``dphidN``: `d(phi)/dN`
        ``t``: time (optional)
        ``eta``: conformal time (optional)

    """

    def __init__(self, K, potential, track_time=False, track_eta=False, verbose=False):
        super(InflationEquationsN, self).__init__(K=K, potential=potential, verbose=verbose)
        self._set_independent_variable('_N')
        self.add_variable('phi', 'dphidN')
        self.track_time = track_time
        self.track_eta = track_eta
        if track_time:
            self.add_variable('t')
        if track_eta:
            self.add_variable('eta')

    def __call__(self, x, y):
        """System of coupled ODEs for underlying variables."""
        H2 = self.H2(x, y)
        dphidN = self.dphidN(x, y)
        dH_H = -dphidN**2 / 2 + self.K * np.exp(-2 * x) / H2

        dy = np.zeros_like(y)
        dy[self.idx['phi']] = dphidN
        dy[self.idx['dphidN']] = -(dH_H + 3) * dphidN - self.dVdphi(x, y) / H2
        if self.track_time:
            dy[self.idx['t']] = 1 / np.sqrt(H2)
        if self.track_eta:
            dy[self.idx['eta']] = np.exp(-x) / np.sqrt(H2)
        return dy

    def H2(self, x, y):
        """Compute the square of the Hubble parameter using the Friedmann equation."""
        return (2 * self.V(x, y) - 6 * self.K * np.exp(-2 * x)) / (6 - self.dphidN(x, y)**2)

    def w(self, x, y):
        """Compute the equation of state parameter."""
        V = self.V(x, y)
        dphidt2 = self.H2(x, y) * self.dphidN(x, y)**2
        p = dphidt2 / 2 - V
        rho = dphidt2 / 2 + V
        return p / rho

    def inflating(self, x, y):
        """Inflation diagnostic for event tracking."""
        return self.V(x, y) - self.H2(x, y) * self.dphidN(x, y)**2

    def sol(self, sol, **kwargs):
        """Post-processing of :func:`scipy.integrate.solve_ivp` solution."""
        sol = super(InflationEquationsN, self).sol(sol, **kwargs)
        return sol
