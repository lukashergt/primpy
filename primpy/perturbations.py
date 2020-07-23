#!/usr/bin/env python
""":mod:`primpy.time.perturbations`: comoving curvature perturbations w.r.t. time `t`."""
from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp
from primpy.units import pi
from primpy.equations import Equations


class PrimordialPowerSpectrum(object):
    """Primordial Power spectrum of curvature perturbations."""

    def __init__(self, background, k, **kwargs):
        self.background = background
        self.k = k
        self.k_iMpc = k / background.a0_Mpc
        vacuum = kwargs.pop('vacuum', ('RST',))
        for vac in vacuum:
            setattr(self, 'P_s_%s' % vac, np.full_like(k, np.nan, dtype=float))
            setattr(self, 'P_t_%s' % vac, np.full_like(k, np.nan, dtype=float))


class Perturbation(ABC):
    """Perturbation for wavenumber `k`."""

    def __init__(self, background, k):
        super(Perturbation, self).__init__()
        self.background = background
        self.k = k
        self.scalar = None
        self.tensor = None

    def oscode_postprocessing(self, oscode_sol, **kwargs):
        """Post-processing for `pyoscode.solve` solution.

        Translate `oscode` dictionary output to `solve_ivp` output with attributes `t` and `y`.

        Parameters
        ----------
            oscode_sol : list
                List [scalar_1, scalar_2, tensor_1, tensor_2] of two independent solutions each
                for both scalar and tensor modes, where each element is a dictionary returned by
                `pyoscode.solve`.
        """
        for m, mode in enumerate([self.scalar, self.tensor]):
            for i, sol in enumerate([mode.one, mode.two]):
                idx = 2 * m + i
                sol.steptype = np.array(oscode_sol[idx]['types'])
                sol.t = np.array(oscode_sol[idx]['t'])
                sol.y = np.vstack((oscode_sol[idx]['sol'], oscode_sol[idx]['dsol']))
                mode.sol(sol)
        self._combine_solutions(**kwargs)

    def _combine_solutions(self, **kwargs):
        vacuum = kwargs.pop('vacuum', ['RST'])
        for mode in [self.scalar, self.tensor]:
            y1 = getattr(mode.one, '%s' % mode.var)
            y2 = getattr(mode.two, '%s' % mode.var)
            dy1 = getattr(mode.one, 'd%s' % mode.var)
            dy2 = getattr(mode.two, 'd%s' % mode.var)
            for vac in vacuum:
                uk_i, duk_i = getattr(mode, 'get_vacuum_ic_%s' % vac)()
                a, b = self._get_coefficients_a_b(uk_i=uk_i, duk_i=duk_i,
                                                  y1_i=y1[0], dy1_i=dy1[0],
                                                  y2_i=y2[0], dy2_i=dy2[0])
                uk_end = a * np.nanmedian(y1[-5:]) + b * np.nanmedian(y2[-5:])
                setattr(mode, 'P_%s_%s' % (mode.tag, vac), np.abs(uk_end)**2 * mode.pps_norm)

    @staticmethod
    def _get_coefficients_a_b(uk_i, duk_i, y1_i, dy1_i, y2_i, dy2_i):
        """Coefficients to a linear combination of 2 solutions."""
        a = (uk_i * dy2_i - duk_i * y2_i) / (y1_i * dy2_i - dy1_i * y2_i)
        b = (uk_i * dy1_i - duk_i * y1_i) / (y2_i * dy1_i - dy2_i * y1_i)
        return a, b


class Mode(Equations, ABC):
    """Template for scalar or tensor modes."""

    def __init__(self, background, k):
        super(Mode, self).__init__()
        self.background = background
        self.k = k
        f, d = self.mukhanov_sasaki_frequency_damping()
        self.ms_frequency = f
        self.ms_damping = d
        self.one = solve_ivp(lambda x, y: y, (0, 0), y0=np.zeros(2))
        self.two = solve_ivp(lambda x, y: y, (0, 0), y0=np.zeros(2))

    @abstractmethod
    def mukhanov_sasaki_frequency_damping(self):
        """Frequency and damping term of the Mukhanov-Sasaki equations."""

    @abstractmethod
    def get_vacuum_ic_RST(self):
        """Get initial conditions for scalar modes for RST vacuum."""


class ScalarMode(Mode, ABC):
    """Template for scalar modes."""

    def __init__(self, background, k, **kwargs):
        super(ScalarMode, self).__init__(background=background, k=k)
        self.var = 'Rk'
        self.tag = 's'
        self.pps_norm = self.k**3 / (2 * pi**2)
        self.add_variable('Rk', 'dRk')
        vacuum = kwargs.pop('vacuum', ('RST',))
        for vac in vacuum:
            setattr(self, 'P_s_%s' % vac, np.nan)


class TensorMode(Mode, ABC):
    """Template for tensor modes."""

    def __init__(self, background, k, **kwargs):
        super(TensorMode, self).__init__(background=background, k=k)
        self.var = 'hk'
        self.tag = 't'
        self.pps_norm = self.k**3 / (2 * pi**2) * 2
        self.add_variable('hk', 'dhk')
        vacuum = kwargs.pop('vacuum', ('RST',))
        for vac in vacuum:
            setattr(self, 'P_t_%s' % vac, np.nan)
