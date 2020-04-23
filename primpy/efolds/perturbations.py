#!/usr/bin/env python
""":mod:`primpy.time.perturbations`: comoving curvature perturbations w.r.t. time `t`."""
import numpy as np
import pyoscode
from primpy.efolds.inflation import InflationEquationsN


class PerturbationEquationsN(InflationEquationsN):
    """Equations for comoving curvature perturbations w.r.t. e-folds `N`.

    Solves the Mukhanov-Sasaki equations for the comoving curvature
    perturbations during inflation for curved and flat universes with
    e-folds `N` of the scale factor as independent variable and for the
    wavenumber `k`.

    Independent variable:
        N: e-folds of the scale factor

    Dependent variables:
        R1: comoving curvature perturbation
        dR1: dR1 / dt
        R2: comoving curvature perturbation
        dR2: dR2 / dt

    """

    def __init__(self, K, potential, k):
        super(PerturbationEquationsN, self).__init__(K, potential, track_eta=False)
        assert K == 0 or abs(K) == 1
        self.k = k
        self.add_variable('R1', 'dR1', 'R2', 'dR2')

    def __call__(self, t, y):
        """System of coupled ODEs (Mukhanov-Sasaki equations) for underlying variables."""
        dy = super(PerturbationEquationsN, self).__call__(t, y)  # Compute background variables

        frequency2, damping = mukhanov_sasaki_frequency_damping_N(k=self.k,
                                                                  K=self.K,
                                                                  a2=np.exp(2 * self.N(t, y)),
                                                                  dphi=self.dphidt(t, y),
                                                                  H=self.H(t, y),
                                                                  dV=self.dVdphi(t, y))

        dy[self.idx['R1']] = self.dR1(t, y)
        dy[self.idx['dR1']] = -2 * damping * self.dR1(t, y) - frequency2 * self.R1(t, y)
        dy[self.idx['R2']] = self.dR2(t, y)
        dy[self.idx['dR2']] = -2 * damping * self.dR2(t, y) - frequency2 * self.R2(t, y)
        return dy

    def sol(self, sol, **kwargs):
        """Post-processing of `solve_ivp` solution."""
        sol.k = self.k
        sol = super(PerturbationEquationsN, self).sol(sol, **kwargs)
        return sol


def mukhanov_sasaki_frequency_damping_N(k, K, a2, dphi, H, dV):
    """Frequency and damping term of the Mukhanov-Sasaki equations.

    Frequency and damping term of the Mukhanov-Sasaki equations for the
    comoving curvature perturbations `R` w.r.t. time `t`, where the e.o.m. is
    written as `ddR + 2 * damping * dR + frequency**2 R = 0`.
    """
    k2 = k**2 - 3 * K + k * (K + 1) * K
    kappa2 = dphi**2 / 2 / H**2 * K + k2
    E = -2 * (3 * H + dV / dphi + K / a2 / H)

    frequency2 = (k2 * (2 * k2 / kappa2 - 1) + K * (k2 / kappa2 * E / H + 1)) / a2
    damping = (k2 / kappa2 * (E + dphi**2 / H) + 3 * H) / 2

    return frequency2, damping


def solve_oscode(sol, k_iMpc=None, k_comov=None):
    if k_iMpc is not None:
        sol.k_iMpc = k_iMpc
        sol.k_comov = sol.k2aH(k_iMpc)
    if k_comov is not None:
        assert k_iMpc is None
        sol.k_comov = k_comov

    frequency2, damping = mukhanov_sasaki_frequency_damping_N(k=sol.k_comov,
                                                              K=sol.K,
                                                              a2=np.exp(2 * sol.N),
                                                              dphi=sol.dphidt,
                                                              H=sol.H,
                                                              dV=sol.potential.dV(sol.phi))
    sol1 = pyoscode.solve(ts=sol.t, ws=np.sqrt(frequency2), gs=damping,
                          ti=sol.t[0], tf=sol.t[-1], x0=1, dx0=0,
                          logw=False, logg=False, rtol=1e-6)
    sol2 = pyoscode.solve(ts=sol.t, ws=np.sqrt(frequency2), gs=damping,
                          ti=sol.t[0], tf=sol.t[-1], x0=0, dx0=sol.k_comov,
                          logw=False, logg=False, rtol=1e-6)
    sol.Rk_1 = sol1['sol']
    sol.Rk_2 = sol2['sol']
    sol.dRk_1 = sol1['dsol']
    sol.dRk_2 = sol2['dsol']
    sol.t_1 = sol1['t']
    sol.t_2 = sol2['t']
    sol.steptype_1 = sol1['types']
    sol.steptype_2 = sol2['types']

    print("done")

    # TODO: continue
