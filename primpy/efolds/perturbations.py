#!/usr/bin/env python
""":mod:`primpy.time.perturbations`: comoving curvature perturbations w.r.t. time `t`."""
import numpy as np
from primpy.perturbations import CurvaturePerturbation


class CurvaturePerturbationN(CurvaturePerturbation):
    """Curvature perturbation for wavenumber `k` w.r.t. e-folds `N=ln(a)`.

    Solves the Mukhanov--Sasaki equations w.r.t. number of e-folds `N` of the
    scale factor `a` for curved universes.

    Input Parameters
    ----------------
        background : Bunch object as returned by `primpy.efolds.inflation.InflationEquationsN.sol`
            Monkey-patched version of the Bunch type usually returned by `solve_ivp`.
        k : float
            wavenumber
    """

    def __init__(self, background, k, mode):
        super(CurvaturePerturbationN, self).__init__(background=background, k=k, mode=mode)
        self._set_independent_variable('N')

    def __call__(self, x, y):
        """Vector of derivatives."""
        raise NotImplementedError("Equations class must define __call__.")

    @staticmethod
    def scalar_mukhanov_sasaki_frequency_damping(background, k):
        """Frequency and damping term of the Mukhanov-Sasaki equations for scalar modes.

        Frequency and damping term of the Mukhanov-Sasaki equations for the
        comoving curvature perturbations `R` w.r.t. e-folds `N`, where the e.o.m. is
        written as `ddR + 2 * damping * dR + frequency**2 R = 0`.
        """
        K = background.K
        a2 = np.exp(2 * background.N)
        H = background.H
        dphidN = background.dphidN
        H2 = H**2
        dV = background.potential.dV(background.phi)
        Omega_K = background.Omega_K

        kappa2 = k**2 + k * K * (K + 1) - 3 * K
        epsilon = dphidN**2 / 2
        xi = Omega_K + epsilon - 3

        damping2 = 2 * kappa2 / (kappa2 + K * epsilon) * (xi - dV / (H2 * dphidN)) - xi
        frequency2 = kappa2 / (a2 * H2) + (damping2 + xi + 1) * Omega_K
        if np.all(frequency2 > 0):
            return np.sqrt(frequency2), damping2 / 2
        else:
            return np.sqrt(frequency2 + 0j), damping2 / 2

    @staticmethod
    def tensor_mukhanov_sasaki_frequency_damping(background, k):
        """Frequency and damping term of the Mukhanov-Sasaki equations for tensor modes.

        Frequency and damping term of the Mukhanov-Sasaki equations for the
        tensor perturbations `h` w.r.t. e-folds `N`, where the e.o.m. is
        written as `ddh + 2 * damping * dh + frequency**2 h = 0`.
        """
        K = background.K
        frequency2 = (k**2 + k * K * (K + 1) + 2 * K) / background.aH**2
        damping2 = 3 - background.dphidN**2 / 2 + K / background.aH**2
        if np.all(frequency2 > 0):
            return np.sqrt(frequency2), damping2 / 2
        else:
            return np.sqrt(frequency2 + 0j), damping2 / 2

    def sol(self, sol, **kwargs):
        """Post-processing for `pyoscode.solve` solution."""
        sol = super(CurvaturePerturbationN, self).sol(sol, **kwargs)
        return sol

    def get_Rk_i(self):
        """Get vacuum initial conditions for curvature perturbation `R_k`."""
        z_i = self.background.a[0] * self.background.dphidN[0]
        return 1 / np.sqrt(2 * self.k) / z_i

    def get_scalar_vacuum_ic_RST(self):
        """Initial conditions for scalar modes for RST vacuum w.r.t. e-folds `N`."""
        a_i = self.background.a[0]
        H_i = self.background.H[0]
        Rk_i = self.get_Rk_i()
        dRk_i = -1j * self.k / (a_i * H_i) * Rk_i
        return Rk_i, dRk_i

    def get_tensor_vacuum_ic_RST(self):
        """Initial conditions for scalar modes for RST vacuum w.r.t. e-folds `N`."""
        a_i = self.background.a[0]
        H_i = self.background.H[0]
        hk_i = 2 / np.sqrt(2 * self.k) / a_i
        dhk_i = -1j * self.k / (a_i * H_i) * hk_i
        return hk_i, dhk_i
