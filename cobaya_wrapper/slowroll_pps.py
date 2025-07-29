"""Slow-roll inflationary primordial power spectrum (PPS) for use with Cobaya."""
import warnings
import numpy as np
from cobaya_wrapper.powerlaw_pps import ExternalPrimordialPowerSpectrum
from primpy.exceptionhandling import PrimpyError, StepSizeError, PrimpyWarning
import primpy.potentials as pp
from primpy.time.inflation import InflationEquationsT as InflationEquations
from primpy.events import InflationEvent
from primpy.initialconditions import SlowRollIC
from primpy.solver import solve


class SlowRollPPS(ExternalPrimordialPowerSpectrum):

    def initialize(self):
        super().initialize()
        self.Pot = pp.InflationaryPotential

    def get_can_support_params(self):
        return {'A_s', 'n_s', 'N_star', 'rho_reh_GeV', 'w_reh', 'phi0', 'p', 'alpha'}

    def get_can_provide_params(self):
        return {'N_star',  # 'phi_star', 'V_star', 'H_star',
                'N_end', 'phi_end', 'V_end', 'H_end',
                'N_reh', 'w_reh', 'DeltaN_reh', 'rho_reh_GeV',
                'A_s', 'n_s', 'n_run', 'n_runrun', 'A_t', 'n_t', 'r'}

    def calculate(self, state, want_derived=True, **params_values_dict):
        N_star = params_values_dict.get('N_star', 75)
        A_s = params_values_dict.get('A_s')
        n_s = params_values_dict.get('n_s')
        rho_reh_GeV = params_values_dict.get('rho_reh_GeV')
        w_reh = params_values_dict.get('w_reh')
        pot_kwargs = {}
        if 'phi0' in params_values_dict.keys():
            phi0 = params_values_dict.get('phi0')
            pot_kwargs.update(phi0=phi0)
        if 'p' in params_values_dict.keys():
            p = params_values_dict.get('p')
            pot_kwargs.update(p=p)
        if 'alpha' in params_values_dict.keys():
            alpha = params_values_dict.get('alpha')
            pot_kwargs.update(alpha=alpha)

        atol = 1e-14
        rtol = 1e-6
        K = 0
        t_eval = np.logspace(5, 12, (12 - 5) * 1000 + 1)
        pot = self.Pot(**pot_kwargs)
        Lambda, _, _ = pot.sr_As2Lambda(A_s=A_s, N_star=N_star, phi_star=None, **pot_kwargs)
        phi_i = pot.sr_N2phi(N=90)  # choose sufficiently high N to accommodate even highest N_star
        if 'phi0' in pot_kwargs and phi_i > phi0:
            phi_i = phi0
        for i in range(11):
            pot = self.Pot(Lambda=Lambda, **pot_kwargs)
            eq = InflationEquations(K=K, potential=pot, track_eta=False)
            ev = [InflationEvent(eq, +1, terminal=False),  # records inflation start
                  InflationEvent(eq, -1, terminal=True)]   # records inflation end
            ic = SlowRollIC(equations=eq, phi_i=phi_i, N_i=0, t_i=t_eval[0])
            b = solve(ic=ic, events=ev, t_eval=t_eval,
                      atol=1e-18, rtol=2.22045e-14, method='DOP853')
            if not b.success:
                raise StepSizeError(b.message)
            with warnings.catch_warnings(action='ignore', category=PrimpyWarning):
                if w_reh is not None and rho_reh_GeV is not None:
                    b.calibrate_scale_factor(calibration_method='reheating',
                                             rho_reh_GeV=rho_reh_GeV, w_reh=w_reh)
                elif N_star is not None and n_s is None:
                    b.calibrate_scale_factor(calibration_method='N_star', N_star=N_star,
                                             rho_reh_GeV=rho_reh_GeV)
                else:
                    N_star = min(N_star, b.N_tot-0.1)
                    b.calibrate_scale_factor(N_star=N_star, rho_reh_GeV=rho_reh_GeV)
                    b.set_ns(n_s=n_s, rho_reh_GeV=rho_reh_GeV, N_star_min=20, N_star_max=N_star)
            # check whether the target A_s is met
            if abs(b.A_s - A_s) < atol + rtol * A_s:
                break  # when the target is met, exit the loop
            elif i < 10:
                # when the target is not met, use the scaling relations between
                # A_s and Lambda to predict a new Lambda
                Lambda = (A_s / b.A_s)**(1 / 4) * Lambda
            else:
                raise PrimpyError("`A_s` shooting failed.")
        if b.w_reh <= -1/3 or b.w_reh >= 1 or b.DeltaN_reh < 0 or b.N_reh > b.N0:
            raise PrimpyError(f"Unrealistic reheating scenario with w_reh={b.w_reh}, "
                              f"DeltaN_reh={b.DeltaN_reh}, N_star={b.N_star}, "
                              f"N_reh={b.N_reh}, and N0={b.N0}.")
        Pks = b.P_s_approx(self.ks)
        Pkt = b.P_t_approx(self.ks)
        state['primordial_scalar_pk'] = {'kmin': self.kmin,
                                         'kmax': self.kmax,
                                         'Pk': Pks,
                                         'log_regular': True}
        state['primordial_tensor_pk'] = {'kmin': self.kmin,
                                         'kmax': self.kmax,
                                         'Pk': Pkt,
                                         'log_regular': True}
        derived_params = self.get_can_provide_params()
        state['derived'] = {derived_param: getattr(b, derived_param)
                            for derived_param in derived_params}

    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']

    def get_primordial_tensor_pk(self):
        return self.current_state['primordial_tensor_pk']


class MonomialSlowRollPPS(SlowRollPPS):

    def initialize(self):
        super().initialize()
        self.Pot = pp.MonomialPotential


class LinearSlowRollPPS(SlowRollPPS):

    def initialize(self):
        super().initialize()
        self.Pot = pp.LinearPotential


class QuadraticSlowRollPPS(SlowRollPPS):

    def initialize(self):
        super().initialize()
        self.Pot = pp.QuadraticPotential


class StarobinskySlowRollPPS(SlowRollPPS):

    def initialize(self):
        super().initialize()
        self.Pot = pp.StarobinskyPotential


class NaturalSlowRollPPS(SlowRollPPS):

    def initialize(self):
        super().initialize()
        self.Pot = pp.NaturalPotential


class DoubleWell2SlowRollPPS(SlowRollPPS):

    def initialize(self):
        super().initialize()
        self.Pot = pp.DoubleWell2Potential


class DoubleWell4SlowRollPPS(SlowRollPPS):

    def initialize(self):
        super().initialize()
        self.Pot = pp.DoubleWell4Potential


class TmodelSlowRollPPS(SlowRollPPS):

    def initialize(self):
        super().initialize()
        self.Pot = pp.TmodelPotential
