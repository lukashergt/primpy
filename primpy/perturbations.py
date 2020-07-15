#!/usr/bin/env python
""":mod:`primpy.time.perturbations`: comoving curvature perturbations w.r.t. time `t`."""
import numpy as np
from scipy.integrate import solve_ivp
from primpy.units import pi
from primpy.equations import Equations


class CurvaturePerturbation(Equations):
    """Curvature perturbation for wavenumber `k`."""

    def __init__(self, background, k, mode):
        super(CurvaturePerturbation, self).__init__()
        self.background = background
        self.k = k
        self.mode = mode
        if mode == 'scalar':
            self.var = 'Rk'
            self.tag = 's'
            self.add_variable('Rk', 'dRk', 'steptype')
        elif mode == 'tensor':
            self.var = 'hk'
            self.tag = 't'
            self.add_variable('hk', 'dhk', 'steptype')
        else:
            raise ValueError("Only scalar or tensor modes allowed, "
                             "but mode=%s was requested." % mode)
        f, d = getattr(self, '%s_mukhanov_sasaki_frequency_damping' % self.mode)(background, k)
        self.ms_frequency = f
        self.ms_damping = d
        self.one = solve_ivp(lambda x, y: y, (0, 0), y0=np.zeros(3))
        self.two = solve_ivp(lambda x, y: y, (0, 0), y0=np.zeros(3))

    def __call__(self, x, y):
        """Vector of derivatives."""
        raise NotImplementedError("Equations class must define __call__.")

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

        y1 = self.one.y[self.idx['%s' % self.var]]
        y2 = self.two.y[self.idx['%s' % self.var]]
        dy1 = self.one.y[self.idx['d%s' % self.var]]
        dy2 = self.two.y[self.idx['d%s' % self.var]]
        norm = self.k**3 / (2 * pi**2) * (2 if self.mode == 'tensor' else 1)
        for vac in ['RST']:
            uk_i, duk_i = getattr(self, 'get_%s_vacuum_ic_%s' % (self.mode, vac))()
            a, b = self._get_coefficients_a_b(uk_i=uk_i, duk_i=duk_i,
                                              y1_i=y1[0], dy1_i=dy1[0],
                                              y2_i=y2[0], dy2_i=dy2[0])
            uk_end = a * y1[-1] + b * y2[-1]
            duk_end = a * dy1[-1] + b * dy2[-1]
            setattr(sol, '%s_%s_end' % (self.var, vac), uk_end)
            setattr(sol, 'd%s_%s_end' % (self.var, vac), duk_end)
            setattr(sol, 'P_%s_%s' % (self.tag, vac), np.abs(uk_end)**2 * norm)
        return sol

    @staticmethod
    def _get_coefficients_a_b(uk_i, duk_i, y1_i, dy1_i, y2_i, dy2_i):
        """Coefficients to a linear combination of 2 solutions."""
        a = (uk_i * dy2_i - duk_i * y2_i) / (y1_i * dy2_i - dy1_i * y2_i)
        b = (uk_i * dy1_i - duk_i * y1_i) / (y2_i * dy1_i - dy2_i * y1_i)
        return a, b
