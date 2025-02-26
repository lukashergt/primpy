"""Power-law primordial power spectrum (PPS) with features for use with Cobaya."""
import numpy as np
from cobaya_wrapper.powerlaw_pps import (ExternalPrimordialPowerSpectrum,
                                         power_law_primordial_scalar_pk,
                                         power_law_primordial_tensor_pk)


def cutoff(ks, k_cut, s_cut):
    return np.exp(-(ks/k_cut)**s_cut)


def log_oscillations(ks, A_log, w_log, p_log, k_pivot):
    lnk = np.log(ks / k_pivot)
    return A_log * np.cos(w_log * lnk + 2*np.pi * p_log)


def lin_oscillations(ks, A_lin, w_lin, p_lin, k_pivot):
    lnk = ks / k_pivot
    return A_lin * np.cos(w_lin * lnk + 2*np.pi * p_lin)


class LogOscillationPPS(ExternalPrimordialPowerSpectrum):

    def get_can_support_params(self):
        return ['A_s', 'n_s', 'n_run', 'n_runrun',
                'A_t', 'n_t', 'n_t_run', 'r',
                'A_log', 'w_log', 'p_log']

    def calculate(self, state, want_derived=True, **params_values_dict):
        A_s = params_values_dict.get('A_s', None)
        n_s = params_values_dict.get('n_s', 1)
        n_run = params_values_dict.get('n_run', 0)
        n_runrun = params_values_dict.get('n_runrun', 0)
        A_t = params_values_dict.get('A_t', 0)
        n_t = params_values_dict.get('n_t', None)
        n_t_run = params_values_dict.get('n_t_run', None)
        r = params_values_dict.get('r', A_t/A_s)
        A_log = params_values_dict.get('A_log', 0)
        w_log = params_values_dict.get('w_log', None)
        p_log = params_values_dict.get('p_log', None)

        if n_t is None:
            # set from inflationary consistency
            if n_t_run:
                raise Exception('n_t_run set but using inflation consistency (n_t=None)')
            n_t = - r / 8.0 * (2.0 - n_s - r / 8.0)
            n_t_run = r / 8.0 * (r / 8.0 + n_s - 1)
        if A_t > 0:
            if r != A_t / A_s:
                raise Exception("mismatch between `r` and `A_t/A_s`")
        elif r > 0:
            A_t = r * A_s

        Pks0 = power_law_primordial_scalar_pk(self.ks, A_s, n_s, n_run, n_runrun, self.k_pivot)
        Pkt0 = power_law_primordial_tensor_pk(self.ks, A_t, n_t, n_t_run, self.k_pivot)
        logosc_Pk = log_oscillations(self.ks, A_log, w_log, p_log, self.k_pivot)
        Pks = Pks0 * (1 + logosc_Pk)
        Pkt = Pkt0 * (1 + logosc_Pk)
        state['primordial_scalar_pk'] = {'kmin': self.kmin,
                                         'kmax': self.kmax,
                                         'Pk': Pks,
                                         'log_regular': True}
        state['primordial_tensor_pk'] = {'kmin': self.kmin,
                                         'kmax': self.kmax,
                                         'Pk': Pkt,
                                         'log_regular': True}

    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']

    def get_primordial_tensor_pk(self):
        return self.current_state['primordial_tensor_pk']


class LinOscillationPPS(ExternalPrimordialPowerSpectrum):

    def get_can_support_params(self):
        return ['A_s', 'n_s', 'n_run', 'n_runrun',
                'A_t', 'n_t', 'n_t_run', 'r',
                'A_lin', 'w_lin', 'p_lin']

    def calculate(self, state, want_derived=True, **params_values_dict):
        A_s = params_values_dict.get('A_s', None)
        n_s = params_values_dict.get('n_s', 1)
        n_run = params_values_dict.get('n_run', 0)
        n_runrun = params_values_dict.get('n_runrun', 0)
        A_t = params_values_dict.get('A_t', 0)
        n_t = params_values_dict.get('n_t', None)
        n_t_run = params_values_dict.get('n_t_run', None)
        r = params_values_dict.get('r', A_t/A_s)
        A_lin = params_values_dict.get('A_lin', 0)
        w_lin = params_values_dict.get('w_lin', None)
        p_lin = params_values_dict.get('p_lin', None)

        if n_t is None:
            # set from inflationary consistency
            if n_t_run:
                raise Exception('n_t_run set but using inflation consistency (n_t=None)')
            n_t = - r / 8.0 * (2.0 - n_s - r / 8.0)
            n_t_run = r / 8.0 * (r / 8.0 + n_s - 1)
        if A_t > 0:
            if r != A_t / A_s:
                raise Exception("mismatch between `r` and `A_t/A_s`")
        elif r > 0:
            A_t = r * A_s

        Pks0 = power_law_primordial_scalar_pk(self.ks, A_s, n_s, n_run, n_runrun, self.k_pivot)
        Pkt0 = power_law_primordial_tensor_pk(self.ks, A_t, n_t, n_t_run, self.k_pivot)
        linosc_Pk = lin_oscillations(self.ks, A_lin, w_lin, p_lin, self.k_pivot)
        Pks = Pks0 * (1 + linosc_Pk)
        Pkt = Pkt0 * (1 + linosc_Pk)
        state['primordial_scalar_pk'] = {'kmin': self.kmin,
                                         'kmax': self.kmax,
                                         'Pk': Pks,
                                         'log_regular': True}
        state['primordial_tensor_pk'] = {'kmin': self.kmin,
                                         'kmax': self.kmax,
                                         'Pk': Pkt,
                                         'log_regular': True}

    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']

    def get_primordial_tensor_pk(self):
        return self.current_state['primordial_tensor_pk']


class CutoffPPS(ExternalPrimordialPowerSpectrum):

    def get_can_support_params(self):
        return ['A_s', 'n_s', 'n_run', 'n_runrun',
                'A_t', 'n_t', 'n_t_run', 'r',
                'k_cut', 's_cut']

    def calculate(self, state, want_derived=True, **params_values_dict):
        A_s = params_values_dict.get('A_s', None)
        n_s = params_values_dict.get('n_s', 1)
        n_run = params_values_dict.get('n_run', 0)
        n_runrun = params_values_dict.get('n_runrun', 0)
        A_t = params_values_dict.get('A_t', 0)
        n_t = params_values_dict.get('n_t', None)
        n_t_run = params_values_dict.get('n_t_run', None)
        r = params_values_dict.get('r', A_t/A_s)
        k_cut = params_values_dict.get('k_cut', 1e-3)
        s_cut = params_values_dict.get('s_cut', 2)

        if n_t is None:
            # set from inflationary consistency
            if n_t_run:
                raise Exception('n_t_run set but using inflation consistency (n_t=None)')
            n_t = - r / 8.0 * (2.0 - n_s - r / 8.0)
            n_t_run = r / 8.0 * (r / 8.0 + n_s - 1)
        if A_t > 0:
            if r != A_t / A_s:
                raise Exception("mismatch between `r` and `A_t/A_s`")
        elif r > 0:
            A_t = r * A_s

        Pks0 = power_law_primordial_scalar_pk(self.ks, A_s, n_s, n_run, n_runrun, self.k_pivot)
        Pkt0 = power_law_primordial_tensor_pk(self.ks, A_t, n_t, n_t_run, self.k_pivot)
        cutoff_Pk = cutoff(self.ks, k_cut, s_cut)
        Pks = Pks0 * (1 - cutoff_Pk)
        Pkt = Pkt0 * (1 - cutoff_Pk)
        state['primordial_scalar_pk'] = {'kmin': self.kmin,
                                         'kmax': self.kmax,
                                         'Pk': Pks,
                                         'log_regular': True}
        state['primordial_tensor_pk'] = {'kmin': self.kmin,
                                         'kmax': self.kmax,
                                         'Pk': Pkt,
                                         'log_regular': True}

    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']

    def get_primordial_tensor_pk(self):
        return self.current_state['primordial_tensor_pk']
