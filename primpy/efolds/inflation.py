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
        * ``phi``: inflaton field
        * ``dphidN``: `d(phi)/dN`
        * ``t``: time (optional)
        * ``eta``: conformal time (optional)

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
        dH_H = self.get_dH_H(N=x, H2=H2, dphi=dphidN, K=self.K)
        dVdphi = self.dVdphi(x, y)

        dy = np.zeros_like(y)
        dy[self.idx['phi']] = dphidN
        dy[self.idx['dphidN']] = self.get_d2phi(H2=H2, dH_H=dH_H, dphi=dphidN, dV=dVdphi)
        if self.track_time:
            dy[self.idx['t']] = 1 / np.sqrt(H2)
        if self.track_eta:
            dy[self.idx['eta']] = np.exp(-x) / np.sqrt(H2)
        return dy

    @staticmethod
    def get_H2(N, dphi, V, K):
        return (2 * V - 6 * K * np.exp(-2 * N)) / (6 - dphi**2)

    @staticmethod
    def get_dH(N, H, dphi, K):
        # here: dH/dN
        return -dphi**2 * H / 2 + K * np.exp(-2 * N) / H

    @staticmethod
    def get_dH_H(N, H2, dphi, K):
        # here: dH/dN / H
        return -dphi**2 / 2 + K * np.exp(-2 * N) / H2

    @staticmethod
    def get_d2H(N, H, dH, dphi, d2phi, K):
        # here: d2H/dN2
        return -d2phi * dphi * H - dphi**2 * dH / 2 - K * np.exp(-2 * N) * (2 * H + dH) / H**2

    @staticmethod
    def get_d3H(N, H, dH, d2H, dphi, d2phi, d3phi, K):
        # here: d3H/dN3
        d3H = (-d3phi*dphi*H - d2phi**2*H - dphi**2*d2H/2 - 2*d2phi*dphi*dH
               + K*np.exp(-2*N) * (4*H-d2H+4*dH+2*dH**2/H) / H**2)
        return d3H

    @staticmethod
    def get_d2phi(H2, dH_H, dphi, dV):
        # here: d2phi/dN2
        return -(dH_H + 3) * dphi - dV / H2

    @staticmethod
    def get_d3phi(H, dH, d2H, dphi, d2phi, dV, d2V):
        # here: d3phi/dN3
        return (-3-dH/H)*d2phi + (-d2H/H - d2V/H**2 + dH**2/H**2)*dphi + 2*dV*dH/H**3

    @staticmethod
    def get_d4phi(H, dH, d2H, d3H, dphi, d2phi, d3phi, dV, d2V, d3V):
        return ((-3 - dH/H)*d3phi
                + (-2*d2H/H - d2V/H**2 + 2*dH**2/H**2)*d2phi
                + (-d3H/H - d3V*dphi/H**2 + 3*d2H*dH/H**2 + 4*d2V*dH/H**3 - 2*dH**3/H**3)*dphi
                + 2*(d2H/H - 3*dH**2/H**2)*dV/H**2)

    def H2(self, x, y):
        """Compute the square of the Hubble parameter using the Friedmann equation."""
        return self.get_H2(N=x, dphi=self.dphidN(x, y), V=self.V(x, y), K=self.K)

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
