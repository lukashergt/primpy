#!/usr/bin/env python
""":mod:`primpy.parameters`: constants and parameters for primpy."""
from primpy.units import c, a_B, Mpc_m, lp_m, Tp_K

# wavenumber at pivot scale in units of [Mpc-1]
K_STAR = 0.05
K_STAR_lp = K_STAR / Mpc_m * lp_m

# hard coded parameters
T_CMB_K = 2.72548  # +- 0.00057, in Kelvin, arXiv:0911.1955
T_CMB_Tp = T_CMB_K / Tp_K
T_nu_TCMB = (4 / 11)**(1/3)  # T_ncdm in CLASS
T_nu_K = T_nu_TCMB * T_CMB_K
T_nu_Tp = T_nu_K / Tp_K
N_eff = 3.044  # equivalent to N_ur in CLASS for massless neutrinos
z_BBN = 1e9  # rough estimate of redshift of Big Bang Nucleosynthesis

# derived parameters
g0 = 2 + 7/8 * 2 * N_eff * T_nu_TCMB**3
rho_gamma0_kg_im3 = a_B * T_CMB_K**4 / c**2  # in SI units
rho_nu0_kg_im3 = 7/8 * N_eff * T_nu_TCMB**4 * rho_gamma0_kg_im3
rho_r0_kg_im3 = rho_gamma0_kg_im3 + rho_nu0_kg_im3
