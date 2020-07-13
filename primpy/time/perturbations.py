#!/usr/bin/env python
""":mod:`primpy.time.perturbations`: comoving curvature perturbations w.r.t. time `t`."""
import numpy as np
from scipy.integrate import solve_ivp
from primpy.units import pi
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
        self.one = solve_ivp(lambda x, y: y, (0, 0), y0=[0, 0, 0])
        self.two = solve_ivp(lambda x, y: y, (0, 0), y0=[0, 0, 0])
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
        K = background.K
        a2 = np.exp(2 * background.N)
        dphidt = background.dphidt
        H = background.H
        dV = background.potential.dV(background.phi)

        kappa2 = k**2 + k * K * (K + 1) - 3 * K
        shared = 2 * kappa2 / (kappa2 + K * dphidt**2 / (2 * H**2))
        terms = dphidt**2 / (2 * H**2) - 3 - dV / (H * dphidt) - K / (a2 * H**2)

        frequency2 = kappa2 / a2 - K / a2 * (1 + shared * terms)
        damping = (3 * H + shared * terms * H) / 2
        if np.all(frequency2 > 0):
            return np.sqrt(frequency2), damping
        else:
            return np.sqrt(frequency2 + 0j), damping

    def sol(self, sol, **kwargs):
        """Post-processing for `pyoscode.solve` solution."""
        sol1 = kwargs.pop('sol1')
        sol2 = kwargs.pop('sol2')
        # translate oscode output to solve_ivp output:
        sol.one.t = sol1['t']
        sol.two.t = sol2['t']
        sol.one.y = np.vstack((sol1['sol'], sol1['dsol'], sol1['types']))
        sol.two.y = np.vstack((sol2['sol'], sol2['dsol'], sol2['types']))
        self.one = super(CurvaturePerturbationT, self).sol(sol.one, **kwargs)
        self.two = super(CurvaturePerturbationT, self).sol(sol.two, **kwargs)

        for key in ['RST']:
            Rk_i, dRk_i = getattr(self, 'get_vacuum_%s' % key)()
            a, b = self._get_coefficients_a_b(Rk_i=Rk_i, dRk_i=dRk_i,
                                              y1_i=self.one.Rk[0], dy1_i=self.one.dRk[0],
                                              y2_i=self.two.Rk[0], dy2_i=self.two.dRk[0])
            setattr(sol, 'Rk_%s_end' % key, a * self.one.Rk[-1] + b * self.two.Rk[-1])
            setattr(sol, 'dRk_%s_end' % key, a * self.one.dRk[-1] + b * self.two.dRk[-1])
            norm = self.k**3 / (2 * pi**2)
            setattr(sol, 'PPS_%s' % key, np.abs(getattr(sol, 'Rk_%s_end' % key))**2 * norm)
        return sol

    def get_Rk_i(self):
        """Get vacuum initial conditions for curvature perturbation `R_k`."""
        a_i = self.background.a[0]
        dphidt_i = self.background.dphidt[0]
        H_i = self.background.H[0]
        z_i = a_i * dphidt_i / H_i
        return 1 / np.sqrt(2 * self.k) / z_i

    def get_vacuum_RST(self):
        """Get vacuum according to the renormalised stress-energy tensor (RST)."""
        a_i = self.background.a[0]
        Rk_i = self.get_Rk_i()
        dRk_i = -1j * self.k / a_i * Rk_i
        return Rk_i, dRk_i

    @staticmethod
    def _get_coefficients_a_b(Rk_i, dRk_i, y1_i, dy1_i, y2_i, dy2_i):
        """Coefficients to a linear combination of 2 solutions."""
        a = (Rk_i * dy2_i - dRk_i * y2_i) / (y1_i * dy2_i - dy1_i * y2_i)
        b = (Rk_i * dy1_i - dRk_i * y1_i) / (y2_i * dy1_i - dy2_i * y1_i)
        return a, b
