#!/usr/bin/env python
""":mod:`primpy.time.perturbations`: comoving curvature perturbations w.r.t. time `t`."""
import numpy as np
from primpy.perturbations import CurvaturePerturbation


class CurvaturePerturbationT(CurvaturePerturbation):
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
        super(CurvaturePerturbationT, self).__init__(background=background, k=k)
        self._set_independent_variable('t')

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
        sol = super(CurvaturePerturbationT, self).sol(sol, **kwargs)
        return sol

    def get_Rk_i(self):
        """Get vacuum initial conditions for curvature perturbation `R_k`."""
        a_i = self.background.a[0]
        dphidt_i = self.background.dphidt[0]
        H_i = self.background.H[0]
        z_i = a_i * dphidt_i / H_i
        return 1 / np.sqrt(2 * self.k) / z_i

    def get_scalar_vacuum_ic_RST(self):
        """Initial conditions for scalar modes for RST vacuum w.r.t. cosmic time `t`."""
        a_i = self.background.a[0]
        Rk_i = self.get_Rk_i()
        dRk_i = -1j * self.k / a_i * Rk_i
        return Rk_i, dRk_i

    def get_tensor_vacuum_ic_RST(self):
        """Initial conditions for tensor modes for RST vacuum w.r.t. cosmic time `t`."""
        a_i = self.background.a[0]
        hk_i = 1 / np.sqrt(2 * self.k) / a_i
        dhk_i = -1j * self.k / a_i * hk_i
        return hk_i, dhk_i
