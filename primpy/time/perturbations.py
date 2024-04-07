#!/usr/bin/env python
""":mod:`primpy.time.perturbations`: curvature perturbations with respect to time `t`."""
import numpy as np
from primpy.perturbations import Perturbation, ScalarMode, TensorMode


class PerturbationT(Perturbation):
    """Curvature perturbation for wavenumber `k` with respect to time `t`.

    Solves the Mukhanov--Sasaki equations w.r.t. cosmic time for curved universes.

    Input Parameters
    ----------------
        background : Bunch object
            Background solution as returned by
            :func:`primpy.time.inflation.InflationEquationsN.sol`.
            Monkey-patched version of the Bunch type usually returned by
            :func:`scipy.integrate.solve_ivp`.
        k : float
            wavenumber

    """

    def __init__(self, background, k, **kwargs):
        super(PerturbationT, self).__init__(background=background, k=k)
        self.scalar = ScalarModeT(background=background, k=k, **kwargs)
        self.tensor = TensorModeT(background=background, k=k, **kwargs)


class ScalarModeT(ScalarMode):
    """Template for scalar modes."""

    def __init__(self, background, k, **kwargs):
        super(ScalarModeT, self).__init__(background=background, k=k, **kwargs)
        self._set_independent_variable('t')

    def __call__(self, x, y):
        """Vector of derivatives."""
        raise NotImplementedError("Equations class must define __call__.")

    def mukhanov_sasaki_frequency_damping(self):
        """Frequency and damping term of the Mukhanov-Sasaki equations for scalar modes.

        Frequency and damping term of the Mukhanov-Sasaki equations for the
        comoving curvature perturbations `R` w.r.t. time `t`, where the e.o.m. is
        written as `ddR + 2 * damping * dR + frequency**2 R = 0`.

        """
        K = self.background.K
        N = self.background.N[self.idx_beg:self.idx_end+1]
        dphidt = self.background.dphidt[self.idx_beg:self.idx_end+1]
        H = self.background.H[self.idx_beg:self.idx_end+1]
        dV = self.background.potential.dV(self.background.phi[self.idx_beg:self.idx_end+1])

        kappa2 = self.k**2 + self.k * K * (K + 1) - 3 * K
        shared = 2 * kappa2 / (kappa2 + K * dphidt**2 / (2 * H**2))
        terms = dphidt**2 / (2 * H**2) - 3 - dV / (H * dphidt) - K * np.exp(-2 * N) / H**2

        frequency2 = kappa2 * np.exp(-2 * N) - K * np.exp(-2 * N) * (1 + shared * terms)
        damping = (3 * H + shared * terms * H) / 2
        if np.all(frequency2 > 0):
            return np.sqrt(frequency2), damping
        else:
            return np.sqrt(frequency2 + 0j), damping

    def get_vacuum_ic_RST(self):
        """Get initial conditions for scalar modes for RST vacuum w.r.t. cosmic time `t`."""
        a_i = np.exp(self.background.N[self.idx_beg])
        dphidt_i = self.background.dphidt[self.idx_beg]
        H_i = self.background.H[self.idx_beg]
        z_i = a_i * dphidt_i / H_i
        Rk_i = 1 / np.sqrt(2 * self.k) / z_i
        dRk_i = -1j * self.k / a_i * Rk_i
        return Rk_i, dRk_i


class TensorModeT(TensorMode):
    """Template for tensor modes."""

    def __init__(self, background, k, **kwargs):
        super(TensorModeT, self).__init__(background=background, k=k, **kwargs)
        self._set_independent_variable('t')

    def __call__(self, x, y):
        """Vector of derivatives."""
        raise NotImplementedError("Equations class must define __call__.")

    def mukhanov_sasaki_frequency_damping(self):
        """Frequency and damping term of the Mukhanov-Sasaki equations for tensor modes.

        Frequency and damping term of the Mukhanov-Sasaki equations for the
        tensor perturbations `h` w.r.t. time `t`, where the e.o.m. is
        written as `ddh + 2 * damping * dh + frequency**2 h = 0`.

        """
        K = self.background.K
        N = self.background.N[self.idx_beg:self.idx_end+1]
        frequency2 = (self.k**2 + self.k * K * (K + 1) + 2 * K) * np.exp(-2 * N)
        damping2 = 3 * self.background.H[self.idx_beg:self.idx_end+1]
        if np.all(frequency2 > 0):
            return np.sqrt(frequency2), damping2 / 2
        else:
            return np.sqrt(frequency2 + 0j), damping2 / 2

    def get_vacuum_ic_RST(self):
        """Get initial conditions for tensor modes for RST vacuum w.r.t. cosmic time `t`."""
        a_i = np.exp(self.background.N[self.idx_beg])
        hk_i = 2 / np.sqrt(2 * self.k) / a_i
        dhk_i = -1j * self.k / a_i * hk_i
        return hk_i, dhk_i
