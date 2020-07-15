#!/usr/bin/env python
""":mod:`primpy.time.perturbations`: comoving curvature perturbations w.r.t. time `t`."""
import numpy as np
from scipy.integrate import solve_ivp
from primpy.units import pi
from primpy.equations import Equations


class CurvaturePerturbation(Equations):
    """Curvature perturbation for wavenumber `k`."""

    def __init__(self, background, k):
        super(CurvaturePerturbation, self).__init__()
        self.background = background
        self.k = k
        self.add_variable('Rk', 'dRk', 'steptype_s')
        # self.add_variable('Rk', 'dRk', 'steptype_s', 'hk', 'dhk', 'steptype_t')
        self.one = solve_ivp(lambda x, y: y, (0, 0), y0=np.zeros(6))
        self.two = solve_ivp(lambda x, y: y, (0, 0), y0=np.zeros(6))
        self.ms_frequency, self.ms_damping = self.mukhanov_sasaki_frequency_damping(background, k)

    def __call__(self, x, y):
        """Vector of derivatives."""
        raise NotImplementedError("Equations class must define __call__.")

    @staticmethod
    def mukhanov_sasaki_frequency_damping(background, k):
        """Frequency and damping term of the Mukhanov-Sasaki equations."""
        raise NotImplementedError("Needs to be implemented for specific independent variable.")

    def sol(self, sol, **kwargs):
        """Post-processing for `pyoscode.solve` solution."""
        one = kwargs.pop('sol1')
        two = kwargs.pop('sol2')
        # translate oscode output to solve_ivp output:
        sol.one.t = one['t']
        sol.two.t = two['t']
        sol.one.y = np.vstack((one['sol'], one['dsol'], one['types']))
        sol.two.y = np.vstack((two['sol'], two['dsol'], two['types']))
        self.one = super(CurvaturePerturbation, self).sol(sol.one, **kwargs)
        self.two = super(CurvaturePerturbation, self).sol(sol.two, **kwargs)

        norm = self.k ** 3 / (2 * pi ** 2)
        # for mode, vac in itertools.product(['scalar', 'tensor'], ['RST']):
        for vac in ['RST']:
            # scalars:
            Rk_i, dRk_i = getattr(self, 'get_scalar_vacuum_ic_%s' % vac)()
            a, b = self._get_coefficients_a_b(Rk_i=Rk_i, dRk_i=dRk_i,
                                              y1_i=self.one.Rk[0], dy1_i=self.one.dRk[0],
                                              y2_i=self.two.Rk[0], dy2_i=self.two.dRk[0])
            setattr(sol, 'Rk_%s_end' % vac, a * self.one.Rk[-1] + b * self.two.Rk[-1])
            setattr(sol, 'dRk_%s_end' % vac, a * self.one.dRk[-1] + b * self.two.dRk[-1])
            setattr(sol, 'P_s_%s' % vac, np.abs(getattr(sol, 'Rk_%s_end' % vac))**2 * norm)
            # # tensors:
            # hk_i, dhk_i = getattr(self, 'get_tensor_vacuum_ic_%s' % vac)()
            # a, b = self._get_coefficients_a_b(Rk_i=hk_i, dRk_i=dhk_i,
            #                                   y1_i=self.one.hk[0], dy1_i=self.one.dhk[0],
            #                                   y2_i=self.two.hk[0], dy2_i=self.two.dhk[0])
            # setattr(sol, 'hk_%s_end' % vac, a * self.one.hk[-1] + b * self.two.hk[-1])
            # setattr(sol, 'dhk_%s_end' % vac, a * self.one.dhk[-1] + b * self.two.dhk[-1])
            # setattr(sol, 'P_t_%s' % vac, np.abs(getattr(sol, 'hk_%s_end' % vac))**2 * norm)
        return sol

    @staticmethod
    def _get_coefficients_a_b(Rk_i, dRk_i, y1_i, dy1_i, y2_i, dy2_i):
        """Coefficients to a linear combination of 2 solutions."""
        a = (Rk_i * dy2_i - dRk_i * y2_i) / (y1_i * dy2_i - dy1_i * y2_i)
        b = (Rk_i * dy1_i - dRk_i * y1_i) / (y2_i * dy1_i - dy2_i * y1_i)
        return a, b
