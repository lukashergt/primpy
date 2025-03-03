"""Power-law primordial power spectrum (PPS) for use with Cobaya."""
import numpy as np
from cobaya.theory import Theory
from primpy.parameters import K_STAR
from primpy.__version__ import __version__


def power_law_primordial_scalar_pk(ks, A_s, n_s, n_run, n_runrun, k_pivot):
    lnk = np.log(ks / k_pivot)
    return A_s * np.exp((n_s - 1) * lnk + n_run / 2 * lnk**2 + n_runrun / 6 * lnk**3)


def power_law_primordial_tensor_pk(ks, A_t, n_t, n_t_run, k_pivot):
    lnk = np.log(ks / k_pivot)
    return A_t * np.exp(n_t * lnk + n_t_run / 2 * lnk**2)


class ExternalPrimordialPowerSpectrum(Theory):
    k_pivot = K_STAR

    def initialize(self):
        # need to provide valid results at wide k range, any that might be used
        super().initialize()
        self.mpi_info(f"Using `primpy` module with version {__version__}.")
        self.kmin = 5e-7
        self.kmax = 5e1
        logkmin = np.log10(self.kmin)
        logkmax = np.log10(self.kmax)
        num_k = int((logkmax-logkmin) * 100 + 1)
        self.ks = np.logspace(logkmin, logkmax, num_k)


class PowerLawPPS(ExternalPrimordialPowerSpectrum):

    # params = {'A_s': None, 'n_s': None, 'n_run': None, 'n_runrun': None,
    #           'A_t': None, 'n_t': None, 'n_t_run': None, 'r': None}

    def get_can_support_params(self):
        return ['A_s', 'n_s', 'n_run', 'n_runrun', 'A_t', 'n_t', 'n_t_run', 'r']

    def calculate(self, state, want_derived=True, **params_values_dict):
        A_s = params_values_dict.get('A_s', None)
        n_s = params_values_dict.get('n_s', 1)
        n_run = params_values_dict.get('n_run', 0)
        n_runrun = params_values_dict.get('n_runrun', 0)
        A_t = params_values_dict.get('A_t', 0)
        n_t = params_values_dict.get('n_t', None)
        n_t_run = params_values_dict.get('n_t_run', 0)
        r = params_values_dict.get('r', A_t/A_s)
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

        Pks = power_law_primordial_scalar_pk(self.ks, A_s, n_s, n_run, n_runrun, self.k_pivot)
        Pkt = power_law_primordial_tensor_pk(self.ks, A_t, n_t, n_t_run, self.k_pivot)
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
