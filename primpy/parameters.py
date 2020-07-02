#!/usr/bin/env python
""":mod:`primpy.parameters`: constants and parameters for primpy."""
from primpy.units import c, a_B

# wavenumber at pivot scale in units of [Mpc-1]
K_STAR = 0.05

# parameters
T_CMB = 2.72548  # +- 0.00057, in Kelvin, arXiv:0911.1955
rho_gamma0_kg_im3 = a_B * T_CMB**4 / c**2  # in SI units
N_eff = 3.046
rho_nu0_kg_im3 = N_eff * 7/8 * (4/11)**(4/3) * rho_gamma0_kg_im3
rho_r0_kg_im3 = rho_gamma0_kg_im3 + rho_nu0_kg_im3
