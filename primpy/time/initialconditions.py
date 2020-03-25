#!/usr/bin/env python
""":mod:`primpy.time.initialconditions`: initial conditions for inflation."""
import numpy as np
from primpy.time.inflation import InflationEquationsT


class ISIC_NiPi(object):
    """Inflation start initial conditions given N_i, phi_i.

    Class for setting up initial conditions at the start of inflation, when
    the curvature density parameter was maximal after kinetic dominance.
    """

    def __init__(self, t_i, N_i, phi_i, K, potential, t_end=1e300):
        self.x_ini = t_i
        self.x_end = t_end
        self.N_i = N_i
        self.phi_i = phi_i
        self.potential = potential
        self.equations = InflationEquationsT(K=K, potential=potential)

    def __call__(self, y0):
        """Background equations of inflation for `N`, `phi` and `dphidt` w.r.t. time `t`."""
        N_i = self.N_i
        phi_i = self.phi_i
        V_i = self.potential.V(phi_i)
        dphidt_i = - np.sqrt(V_i)
        self.aH_i = np.sqrt(V_i / 2 * np.exp(2 * N_i) - self.equations.K)
        self.Omega_ki = -self.equations.K / self.aH_i**2

        self.equations.potential = self.potential
        y0[self.equations.idx['N']] = N_i
        y0[self.equations.idx['phi']] = phi_i
        y0[self.equations.idx['dphidt']] = dphidt_i
