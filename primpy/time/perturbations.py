"""Curvature perturbations with respect to time `t`."""
import numpy as np
from primpy.time.inflation import InflationEquationsT as Eq
from primpy.perturbations import Perturbation, ScalarMode, TensorMode


class PerturbationT(Perturbation):
    """Curvature perturbation for wavenumber `k` with respect to time `t`.

    Solves the Mukhanov--Sasaki equations w.r.t. cosmic time for curved universes.

    Parameters
    ----------
    background : Bunch object same as returned by :func:`scipy.integrate.solve_ivp`
        Background solution as returned by :func:`primpy.time.inflation.InflationEquationsT.sol`.
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
        if 'num_eval' in kwargs and kwargs['num_eval'] > 0:
            self.t_eval = np.logspace(np.log10(self.background.t[self.idx_beg]),
                                      np.log10(self.background.t[self.idx_end]),
                                      kwargs['num_eval'])
            self.x_eval = self.t_eval
        else:
            self.x_eval = None

    def __call__(self, x, y):
        """Vector of derivatives."""
        raise NotImplementedError("Equations class must define __call__.")

    def mukhanov_sasaki_frequency_damping(self):
        """Frequency and damping term of the Mukhanov-Sasaki equations for scalar modes.

        Frequency and damping term of the Mukhanov-Sasaki equations for the
        comoving curvature perturbations `R` w.r.t. time `t`, where the e.o.m. is
        written as `ddR + 2 * damping * dR + frequency**2 * R = 0`.

        """
        K = self.background.K
        N = self.background._N[self.idx_beg:self.idx_end+1]
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

    def get_vacuum_ic_k(self):
        """Get initial conditions for scalar modes for HD approximation w.r.t. cosmic time `t`."""
        N_i = self.background._N[self.idx_beg]
        a_i = np.exp(N_i)
        H_i = self.background.H[self.idx_beg]
        phi_i = self.background.phi[self.idx_beg]
        dV_i = self.background.potential.dV(phi_i)
        dphi_i = self.background.dphidt[self.idx_beg]
        dH_H_i = Eq.get_dH_H(N=N_i, H2=H_i**2, dphi=dphi_i, K=self.background.K)
        d2phi_i = Eq.get_d2phi(H2=H_i**2, dH_H=dH_H_i, dphi=dphi_i, dV=dV_i)

        z_i = a_i * dphi_i / H_i
        dz_z_i = H_i + d2phi_i / dphi_i - dH_H_i

        Rk_i = 1 / np.sqrt(2 * self.k) / z_i
        dRk_i = (-1j * self.k / a_i - dz_z_i) * Rk_i
        return Rk_i, dRk_i

    def get_vacuum_ic_HD(self):
        """Get initial conditions for scalar modes for HD vacuum w.r.t. cosmic time `t`."""
        N_i = self.background._N[self.idx_beg]
        a_i = np.exp(N_i)
        H_i = self.background.H[self.idx_beg]
        phi_i = self.background.phi[self.idx_beg]
        dV_i = self.background.potential.dV(phi_i)
        d2V_i = self.background.potential.d2V(phi_i)
        dphi_i = self.background.dphidt[self.idx_beg]
        dH_i = Eq.get_dH(N=N_i, H=H_i, dphi=dphi_i, K=self.background.K)
        dH_H_i = Eq.get_dH_H(N=N_i, H2=H_i**2, dphi=dphi_i, K=self.background.K)
        d2phi_i = Eq.get_d2phi(H2=H_i**2, dH_H=dH_H_i, dphi=dphi_i, dV=dV_i)
        d2H_i = Eq.get_d2H(N=N_i, H=H_i, dH=dH_i, dphi=dphi_i, d2phi=d2phi_i, K=self.background.K)
        d3phi_i = Eq.get_d3phi(H=H_i, dH=dH_i, d2H=d2H_i, dphi=dphi_i, d2phi=d2phi_i,
                               dV=dV_i, d2V=d2V_i)

        z_i = a_i * dphi_i / H_i
        dz_z_i = H_i + d2phi_i / dphi_i - dH_H_i
        d2z_z_i = (H_i**2 + 2 * H_i * d2phi_i / dphi_i + d3phi_i / dphi_i - dH_i
                   - d2H_i / H_i - 2 * d2phi_i * dH_H_i / dphi_i + 2 * dH_H_i**2)
        wk_i = a_i * np.sqrt(self.k**2/a_i**2 - H_i * dz_z_i - d2z_z_i)

        Rk_i = 1 / np.sqrt(2 * wk_i) / z_i
        dRk_i = (-1j * wk_i / a_i - dz_z_i) * Rk_i
        return Rk_i, dRk_i

    def get_vacuum_ic_RST(self):
        """Get initial conditions for scalar modes for RST vacuum w.r.t. cosmic time `t`."""
        a_i = np.exp(self.background._N[self.idx_beg])
        dphi_i = self.background.dphidt[self.idx_beg]
        H_i = self.background.H[self.idx_beg]
        z_i = a_i * dphi_i / H_i
        Rk_i = 1 / np.sqrt(2 * self.k) / z_i
        dRk_i = -1j * self.k / a_i * Rk_i
        return Rk_i, dRk_i


class TensorModeT(TensorMode):
    """Template for tensor modes."""

    def __init__(self, background, k, **kwargs):
        super(TensorModeT, self).__init__(background=background, k=k, **kwargs)
        self._set_independent_variable('t')
        if 'num_eval' in kwargs and kwargs['num_eval'] > 0:
            self.t_eval = np.logspace(np.log10(self.background.t[self.idx_beg]),
                                      np.log10(self.background.t[self.idx_end]),
                                      kwargs['num_eval'])
            self.x_eval = self.t_eval
        else:
            self.x_eval = None

    def __call__(self, x, y):
        """Vector of derivatives."""
        raise NotImplementedError("Equations class must define __call__.")

    def mukhanov_sasaki_frequency_damping(self):
        """Frequency and damping term of the Mukhanov-Sasaki equations for tensor modes.

        Frequency and damping term of the Mukhanov-Sasaki equations for the
        tensor perturbations `h` w.r.t. time `t`, where the e.o.m. is
        written as `ddh + 2 * damping * dh + frequency**2 * h = 0`.

        """
        K = self.background.K
        N = self.background._N[self.idx_beg:self.idx_end+1]
        frequency2 = (self.k**2 + self.k * K * (K + 1) + 2 * K) * np.exp(-2 * N)
        damping2 = 3 * self.background.H[self.idx_beg:self.idx_end+1]
        if np.all(frequency2 > 0):
            return np.sqrt(frequency2), damping2 / 2
        else:
            return np.sqrt(frequency2 + 0j), damping2 / 2

    def get_vacuum_ic_k(self):
        """Get initial conditions for tensor modes for HD approximation w.r.t. cosmic time `t`."""
        a_i = np.exp(self.background._N[self.idx_beg])
        H_i = self.background.H[self.idx_beg]
        hk_i = 2 / np.sqrt(2 * self.k) / a_i
        dhk_i = (-1j * self.k / a_i - H_i) * hk_i
        return hk_i, dhk_i

    def get_vacuum_ic_HD(self):
        """Get initial conditions for tensor modes for HD vacuum w.r.t. cosmic time `t`."""
        N_i = self.background._N[self.idx_beg]
        a_i = np.exp(N_i)
        H_i = self.background.H[self.idx_beg]
        dphi_i = self.background.dphidt[self.idx_beg]
        dH_i = Eq.get_dH(N=N_i, H=H_i, dphi=dphi_i, K=self.background.K)
        wk_i = a_i * np.sqrt(self.k**2/a_i**2 - 2 * H_i**2 - dH_i)
        hk_i = 2 / np.sqrt(2 * wk_i) / a_i
        dhk_i = (-1j * wk_i / a_i - H_i) * hk_i
        return hk_i, dhk_i

    def get_vacuum_ic_RST(self):
        """Get initial conditions for tensor modes for RST vacuum w.r.t. cosmic time `t`."""
        a_i = np.exp(self.background._N[self.idx_beg])
        hk_i = 2 / np.sqrt(2 * self.k) / a_i
        dhk_i = -1j * self.k / a_i * hk_i
        return hk_i, dhk_i
