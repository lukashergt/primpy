#!/usr/bin/env python
""":mod:`primpy.time.perturbations`: comoving curvature perturbations w.r.t. time `t`."""
import numpy as np
from primpy.equations import Equations


class CurvaturePerturbationT(Equations):
    """Curvature perturbation for wavenumber `k` w.r.t. time `t`.

    Solves the Mukhanov--Sasaki equations w.r.t. cosmic time for curved universes.

    Input Parameters
    ----------------
        background : Bunch object as returned by `primpy.time.inflation.InflationEquationsT.sol`
            Monkey-patched version of the Bunch type usually returned by `solve_ivp`.
        k : float
            wavenumber
    """

    def __init__(self, background, k):
        super(CurvaturePerturbationT, self).__init__()
        self.background = background
        self.k = k
        self._set_independent_variable('t')
        self.add_variable('Rk', 'dRk', 'steptype')
        self.one = None
        self.two = None
        self.ms_frequency, self.ms_damping = self.mukhanov_sasaki_frequency_damping(background, k)

    def __call__(self, x, y):
        """Vector of derivatives."""
        raise NotImplementedError("Equations class must define __call__.")

    @staticmethod
    def mukhanov_sasaki_frequency_damping(background, k):
        """Frequency and damping term of the Mukhanov-Sasaki equations.

        Frequency and damping term of the Mukhanov-Sasaki equations for the
        comoving curvature perturbations `R` w.r.t. time `t`, where the e.o.m. is
        written as `ddR + 2 * damping * dR + frequency**2 R = 0`.
        """
        k = k
        K = background.K
        a2 = np.exp(2 * background.N)
        dphi = background.dphidt
        H = background.H
        dV = background.potential.dV(background.phi)

        k2 = k**2 - 3 * K + k * (K + 1) * K
        kappa2 = dphi**2 / 2 / H**2 * K + k2
        E = -2 * (3 * H + dV / dphi + K / a2 / H)

        frequency2 = (k2 * (2 * k2 / kappa2 - 1) + K * (k2 / kappa2 * E / H + 1)) / a2
        damping = (k2 / kappa2 * (E + dphi**2 / H) + 3 * H) / 2
        return np.sqrt(frequency2), damping

    def sol(self, sol, **kwargs):
        sol1 = kwargs.pop('sol1')
        sol2 = kwargs.pop('sol2')
        # translate oscode output to solve_ivp output:
        sol1.t = sol1['t']
        sol2.t = sol2['t']
        sol1.y = np.vstack((sol1['sol'], sol1['dsol'], sol1['types']))
        sol2.y = np.vstack((sol2['sol'], sol2['dsol'], sol2['types']))
        self.one = super(CurvaturePerturbationT, self).sol(sol1, **kwargs)
        self.two = super(CurvaturePerturbationT, self).sol(sol2, **kwargs)

        for key in ['RST']:
            Rk_i, dRk_i = getattr(self, 'get_vacuum_%s' % key)()
            a, b = self._get_coefficients_a_b(Rk_i=Rk_i, dRk_i=dRk_i,
                                              y1_i=self.one.Rk[0], dy1_i=self.one.dRk[0],
                                              y2_i=self.two.Rk[0], dy2_i=self.two.dRk[0])
            setattr(sol, 'Rk_%s' % key, a * self.one.Rk + b * self.two.Rk)
            setattr(sol, 'dRk_%s' % key, a * self.one.dRk + b * self.two.dRk)
            setattr(sol, 'Rk_%s_end' % key, getattr(sol, 'Rk_%s' % key)[-1])
            setattr(sol, 'dRk_%s_end' % key, getattr(sol, 'dRk_%s' % key)[-1])

        return sol

    def get_Rk_i(self):
        """Set vacuum initial conditions for `R_k`."""
        a_i = self.background.a[0]
        dphidt_i = self.background.dphidt[0]
        H_i = self.background.H[0]
        z_i = a_i * dphidt_i / H_i
        return 1 / np.sqrt(2 * self.k) / z_i

    def get_vacuum_RST(self):
        a_i = self.background.a[0]
        Rk_i = self.get_Rk_i()
        dRk_i = -1j * self.k / a_i * Rk_i
        return Rk_i, dRk_i

    @staticmethod
    def _get_coefficients_a_b(Rk_i, dRk_i, y1_i, dy1_i, y2_i, dy2_i):
        a = (Rk_i * dy2_i - dRk_i * y2_i) / (y1_i * dy2_i - dy1_i * y2_i)
        b = (Rk_i * dy1_i - dRk_i * y1_i) / (y2_i * dy1_i - dy2_i * y1_i)
        return a, b
