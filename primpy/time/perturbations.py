#!/usr/bin/env python
""":mod:`primpy.time.perturbations`: comoving curvature perturbations w.r.t. time `t`."""
import numpy as np
import pyoscode
from primpy.time.inflation import InflationEquationsT


class PerturbationEquationsT(InflationEquationsT):
    """Equations for comoving curvature perturbations w.r.t. time `t`.

    Solves the Mukhanov-Sasaki equations for the comoving curvature
    perturbations during inflation for curved and flat universes with cosmic
    time `t` as independent variable and for the wavenumber `k`.

    Independent variable:
        t: cosmic time

    Dependent variables:
        R1: comoving curvature perturbation
        dR1: dR1 / dt
        R2: comoving curvature perturbation
        dR2: dR2 / dt

    """

    def __init__(self, K, potential, k):
        super(PerturbationEquationsT, self).__init__(K, potential, track_eta=False)
        assert K == 0 or abs(K) == 1
        self.k = k
        self.add_variable('R1', 'dR1', 'R2', 'dR2')

    def __call__(self, t, y):
        """System of coupled ODEs (Mukhanov-Sasaki equations) for underlying variables."""
        dy = super(PerturbationEquationsT, self).__call__(t, y)  # Compute background variables

        frequency2, damping = mukhanov_sasaki_frequency_damping(k=self.k,
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
        sol = super(PerturbationEquationsT, self).sol(sol, **kwargs)
        return sol


def mukhanov_sasaki_frequency_damping(k, K, a2, dphi, H, dV):
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


class CurvaturePerturbationT(object):
    def __init__(self, background, k, one, two):
        self.one = one
        self.two = two
        y1_i = one['sol'][0]
        dy1_i = one['dsol'][0]
        y2_i = two['sol'][0]
        dy2_i = two['dsol'][0]
        for key in ['BD', 'HD', 'RST']:
            R_ki, dR_ki = get_vacuum_ic(background=background, k=k, vacuumtype=key)
            a, b = _get_coefficients_a_b(R_ki=R_ki, dR_ki=dR_ki,
                                         y1_i=y1_i, dy1_i=dy1_i,
                                         y2_i=y2_i, dy2_i=dy2_i)
            setattr(self, 'R_k_' + key + '_end', a * one['sol'][-1] + b * two['sol'][-1])
            setattr(self, 'dR_k_' + key + '_end', a * one['dsol'][-1] + b * two['dsol'][-1])

    def __call__(self, *args, **kwargs):
        pass


def solve_oscode(background, k):
    frequency2, damping = mukhanov_sasaki_frequency_damping(
        k=k,
        K=background.K,
        a2=np.exp(2 * background.N),
        dphi=background.dphidt,
        H=background.H,
        dV=background.potential.dV(background.phi)
    )
    # initial conditions
    y1_i = 1
    dy1_i = 0
    y2_i = 0
    dy2_i = k
    one = pyoscode.solve(ts=background.t, ws=np.sqrt(frequency2), gs=damping,
                         ti=background.t[0], tf=background.t[-1], x0=y1_i, dx0=dy1_i,
                         logw=False, logg=False, rtol=1e-6)
    two = pyoscode.solve(ts=background.t, ws=np.sqrt(frequency2), gs=damping,
                         ti=background.t[0], tf=background.t[-1], x0=y2_i, dx0=dy2_i,
                         logw=False, logg=False, rtol=1e-6)

    sol = CurvaturePerturbationT(background, k, one, two)
    sol.rk_1 = one['sol']
    sol.rk_2 = two['sol']
    sol.drk_1 = one['dsol']
    sol.drk_2 = two['dsol']
    sol.t_1 = one['t']
    sol.t_2 = two['t']
    sol.steptypes_1 = one['types']
    sol.steptypes_2 = two['types']

    sol.Rk_1_BD = background.Rk_1 / np.sqrt(2 * k) / z_i
    sol.Rk_2_BD = background.Rk_2 / np.sqrt(2 * k) / z_i
    sol.dRk_1_BD = background.dRk_1 * (-1j * k / a_i - dz_z_i) * background.Rk_1_BD
    sol.dRk_2_BD = background.dRk_2 * (-1j * k / a_i - dz_z_i) * background.Rk_2_BD

    return sol

    print("done")

    # TODO: continue


def _get_coefficients_a_b(R_ki, dR_ki, y1_i, dy1_i, y2_i, dy2_i):
    a = (R_ki * dy2_i - dR_ki * y2_i) / (y1_i * dy2_i - dy1_i * y2_i)
    b = (R_ki * dy1_i - dR_ki * y1_i) / (y2_i * dy1_i - dy2_i * y1_i)
    return a, b


def get_Rk_i(background, k):
    """Set vacuum initial conditions for `R_k`."""
    z_i = background.a[0] * background.dphidt[0] / background.H[0]
    return 1 / np.sqrt(2 * k) / z_i


def get_bunch_davies_vacuum(background, k):
    Rk_i = get_Rk_i(background=background, k=k)
    return Rk_i, dRk_i


def get_hamiltonian_diagonalisation_vacuum(background, k):
    Rk_i = get_Rk_i(background=background, k=k)
    return Rk_i, dRk_i


def get_renormalised_stress_energy_tensor_vacuum(background, k):
    Rk_i = get_Rk_i(background=background, k=k)
    return Rk_i, dRk_i


def get_vacuum_ic(background, k, vacuumtype):
    if vacuumtype == 'BD':
        return get_bunch_davies_vacuum(background=background, k=k)
    elif vacuumtype == 'HD':
        return get_hamiltonian_diagonalisation_vacuum(background=background, k=k)
    elif vacuumtype == 'RST':
        return get_renormalised_stress_energy_tensor_vacuum(background=background, k=k)
    else:
        raise NotImplementedError("%s vacuum not implemented. The only implemented vacua are: "
                                  "'BD' : Bunch Davies vacuum, "
                                  "'HD' : Hamiltonian diagonalisation vacuum, "
                                  "'RST' : renormalised stress-energy tensor vacuum." % vacuumtype)
