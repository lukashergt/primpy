"""Differential equations for inflation w.r.t. time `t`."""
import numpy as np
from primpy.inflation import InflationEquations


class InflationEquationsT(InflationEquations):
    """Background equations during inflation w.r.t. time `t`.

    Solves background variables in cosmic time for curved and flat universes
    using the Klein-Gordon and Friedmann equations.

    Independent variable:
        ``t``: cosmic time

    Dependent variables:
        * ``_N``: number of e-folds
        * ``phi``: inflaton field
        * ``dphidt``: `d(phi)/dt`
        * ``eta``: conformal time (optional)

    """

    def __init__(self, K, potential, track_eta=False, verbose=False):
        super(InflationEquationsT, self).__init__(K=K, potential=potential, verbose=verbose)
        self._set_independent_variable('t')
        self.add_variable('phi', 'dphidt', '_N')
        self.track_eta = track_eta
        if track_eta:
            self.add_variable('eta')

    def __call__(self, x, y):
        """System of coupled ODEs for underlying variables."""
        N = self._N(x, y)
        H2 = self.H2(x, y)
        H = np.sqrt(H2)
        dphidt = self.dphidt(x, y)
        dVdphi = self.dVdphi(x, y)

        dy = np.zeros_like(y)
        dy[self.idx['phi']] = dphidt
        dy[self.idx['dphidt']] = self.get_d2phi(H2=H2, dH_H=None, dphi=dphidt, dV=dVdphi)
        dy[self.idx['_N']] = H
        if self.track_eta:
            dy[self.idx['eta']] = np.exp(-N)
        return dy

    @staticmethod
    def get_H2(N, dphi, V, K):  # noqa: D102
        return (dphi**2 / 2 + V) / 3 - K * np.exp(-2 * N)

    @staticmethod
    def get_dH(N, H, dphi, K):  # noqa: D102
        # here: dH/dt
        return -dphi**2 / 2 + K * np.exp(-2 * N)

    @staticmethod
    def get_dH_H(N, H2, dphi, K):  # noqa: D102
        # here: dH/dt
        H = np.sqrt(H2)
        return -dphi**2 / (2 * H) + K * np.exp(-2 * N) / H

    @staticmethod
    def get_d2H(N, H, dH, dphi, d2phi, K):  # noqa: D102
        # here: d2H/dt2
        return -d2phi * dphi - 2 * K * H * np.exp(-2 * N)

    @staticmethod
    def get_d3H(N, H, dH, d2H, dphi, d2phi, d3phi, K):  # noqa: D102
        # here: d3H/dt3
        return - d3phi * dphi - d2phi**2 + 2 * K * (2 * H**2 - dH) * np.exp(-2 * N)

    @staticmethod
    def get_d2phi(H2, dH_H, dphi, dV):  # noqa: D102
        # here: d2phi/dt2
        H = np.sqrt(H2)
        return -3 * H * dphi - dV

    @staticmethod
    def get_d3phi(H, dH, d2H, dphi, d2phi, dV, d2V):  # noqa: D102
        return -3 * H * d2phi - d2V * dphi + 3 * dphi**3 / 2

    @staticmethod
    def get_d4phi(H, dH, d2H, d3H, dphi, d2phi, d3phi, dV, d2V, d3V):  # noqa: D102
        return -3 * H * d3phi - d2V * d2phi - d3V * dphi**2 - 3 * d2H * dphi - 6 * d2phi * dH

    def H2(self, x, y):  # noqa: D102
        _N = self._N(x, y)
        V = self.V(x, y)
        dphi = self.dphidt(x, y)
        return self.get_H2(N=_N, dphi=dphi, V=V, K=self.K)

    def w(self, x, y):  # noqa: D102
        V = self.V(x, y)
        dphidt_2 = self.dphidt(x, y)**2
        p = dphidt_2 / 2 - V
        rho = dphidt_2 / 2 + V
        return p / rho

    def inflating(self, x, y):  # noqa: D102
        return self.V(x, y) - self.dphidt(x, y)**2

    def sol(self, sol, **kwargs):  # noqa: D102
        sol = super(InflationEquationsT, self).sol(sol, **kwargs)
        return sol
