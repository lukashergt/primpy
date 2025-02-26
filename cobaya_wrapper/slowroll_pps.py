"""Slow-roll inflationary primordial power spectrum (PPS) for use with Cobaya."""
import numpy as np
from cobaya_wrapper.powerlaw_pps import ExternalPrimordialPowerSpectrum
from primpy.exceptionhandling import PrimpyError
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
        return {'A_s', 'n_s', 'N_star', 'rho_reh_GeV'}

    def get_can_provide_params(self):
        return {'N_star',  # 'phi_star', 'V_star', 'H_star',
                'N_end', 'phi_end', 'V_end', 'H_end',
                'N_reh', 'w_reh', 'DeltaN_reh',
                'A_s', 'n_s', 'n_run', 'n_runrun', 'A_t', 'n_t', 'r'}

    def calculate(self, state, want_derived=True, **params_values_dict):
        N_star = params_values_dict.get('N_star', 90)
        A_s = params_values_dict.get('A_s')
        n_s = params_values_dict.get('n_s')
        rho_reh_GeV = params_values_dict.get('rho_reh_GeV')

        atol = 1e-14
        rtol = 1e-6
        K = 0
        t_eval = np.logspace(5, 8, (8-5)*1000+1)
        Lambda, phi_star, N_star = self.Pot.sr_As2Lambda(A_s=A_s, N_star=N_star, phi_star=None)
        for i in range(11):
            pot = self.Pot(Lambda=Lambda)
            eq = InflationEquations(K=K, potential=pot, track_eta=False)
            ev = [InflationEvent(eq, +1, terminal=False),  # records inflation start
                  InflationEvent(eq, -1, terminal=True)]   # records inflation end
            ic = SlowRollIC(equations=eq, phi_i=phi_star*1.1, N_i=0, t_i=t_eval[0])
            b = solve(ic=ic, events=ev, t_eval=t_eval,
                      atol=1e-18, rtol=2.22045e-14, method='DOP853')
            b.calibrate_scale_factor(N_star=N_star, rho_reh_GeV=rho_reh_GeV)
            if n_s is not None:
                b.set_ns(n_s=n_s, rho_reh_GeV=rho_reh_GeV)
            # check whether the target A_s is met
            if abs(b.A_s - A_s) < atol + rtol * A_s:
                break  # when the target is met, exit the loop
            elif i < 10:
                # when the target is not met, use the scaling relations between
                # A_s and Lambda to predict a new Lambda
                Lambda = (A_s / b.A_s)**(1 / 4) * Lambda
            else:
                raise PrimpyError("`A_s` shooting failed.")
        if b.w_reh <= -1/3 or b.w_reh >= 1 or b.DeltaN_reh < 0:
            raise PrimpyError(f"Unrealistic reheating scenario with w_reh={b.w_reh} and "
                              f"DeltaN_reh={b.DeltaN_reh}.")
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


class StarobinskySlowRollPPS(SlowRollPPS):

    def initialize(self):
        super().initialize()
        self.Pot = pp.StarobinskyPotential
