#!/usr/bin/env python
""":mod:`primpy.initialconditions`: initial conditions for inflation."""
import warnings
import numpy as np
from scipy.optimize import root_scalar
from primpy.time.inflation import InflationEquationsT
from primpy.efolds.inflation import InflationEquationsN
from primpy.solver import solve
from primpy.events import InflationEvent, CollapseEvent


# noinspection PyPep8Naming
class InflationStartIC(object):
    """Inflation start initial conditions given N_i, phi_i.

    Class for setting up initial conditions at the start of inflation, when
    the curvature density parameter was maximal after kinetic dominance.
    """

    def __init__(self, equations, phi_i, t_i=None, eta_i=None, x_end=1e300, **kwargs):
        self.equations = equations
        self.phi_i = phi_i
        self.t_i = t_i
        self.eta_i = eta_i
        self.x_end = x_end

        self.V_i = equations.potential.V(self.phi_i)
        if 'N_i' in kwargs:
            assert 'Omega_Ki' not in kwargs, "Only either N_i or Omega_Ki should be specified. " \
                                             "The other will be inferred."
            self.N_i = kwargs.pop('N_i')
            self.ic_input_param = {'N_i': self.N_i}
            assert self.V_i / 2 * np.exp(2 * self.N_i) - equations.K > 0, \
                ("V_i / 2 * exp(2 N_i) - 1 = %s < 0 but needs to be > 0. "
                 "Increase either N_i or phi_i." % (self.V_i / 2 * np.exp(2 * self.N_i) - 1))
            self.aH_i = np.sqrt(self.V_i / 2 * np.exp(2 * self.N_i) - equations.K)
            self.Omega_Ki = -equations.K / self.aH_i**2
        elif 'Omega_Ki' in kwargs:
            assert 'N_i' not in kwargs, "Only either N_i or Omega_Ki should be specified. " \
                                        "The other will be inferred."
            self.Omega_Ki = kwargs.pop('Omega_Ki')
            self.ic_input_param = {'Omega_Ki': self.Omega_Ki}
            self.N_i = np.log(2 * equations.K / self.V_i * (1 - 1 / self.Omega_Ki)) / 2
            self.aH_i = np.sqrt(-equations.K / self.Omega_Ki)
        else:
            raise IOError("Need to specify either N_i or Omega_Ki.")
        self.H_i = np.sqrt(self.V_i / 2 - equations.K * np.exp(-2 * self.N_i))

    def __call__(self, y0, **ivp_kwargs):
        """Set background equations of inflation for `N`, `phi` and `dphi`."""
        if isinstance(self.equations, InflationEquationsT):
            self.x_ini = self.t_i
            self.x_end = self.x_end
            self.dphidt_i = -np.sqrt(self.V_i)
            y0[self.equations.idx['dphidt']] = self.dphidt_i
            y0[self.equations.idx['N']] = self.N_i
        elif isinstance(self.equations, InflationEquationsN):
            self.x_ini = self.N_i
            self.x_end = self.x_end
            self.dphidN_i = -np.sqrt(self.V_i) / self.H_i
            y0[self.equations.idx['dphidN']] = self.dphidN_i
            if self.equations.track_time:
                assert self.t_i is not None, ("`track_time=%s`, but `t_i=%s`."
                                              % (self.equations.track_time, self.t_i))
                y0[self.equations.idx['t']] = self.t_i
        else:
            raise NotImplementedError("`equations` has to be either of type `InflationEquationsT`"
                                      "or of type `InflationEquationsN`, but type `%s` was given."
                                      % type(self.equations))
        y0[self.equations.idx['phi']] = self.phi_i
        if self.equations.track_eta:
            assert self.eta_i is not None, ("`track_eta=%s`, but `eta_i=%s`."
                                            % (self.equations.track_eta, self.eta_i))
            y0[self.equations.idx['eta']] = self.eta_i


# noinspection PyPep8Naming
class ISIC_NiNt(InflationStartIC):
    """Inflation start initial conditions given potential mass/Lambda, N_tot, and N_i."""

    def __init__(self, equations, N_tot, phi_i_bracket, t_i=None, eta_i=None,
                 x_end=1e300, verbose=False, **kwargs):
        super(ISIC_NiNt, self).__init__(equations=equations,
                                        phi_i=phi_i_bracket[-1],
                                        t_i=t_i,
                                        eta_i=eta_i,
                                        x_end=x_end,
                                        **kwargs)
        self.N_tot = N_tot
        self.phi_i_bracket = phi_i_bracket
        self.verbose = verbose

    def __call__(self, y0, **ivp_kwargs):
        """Set background equations of inflation optimizing for `N_tot`."""
        events = [InflationEvent(self.equations, direction=+1, terminal=False),
                  InflationEvent(self.equations, direction=-1, terminal=True),
                  CollapseEvent(self.equations)]

        def phii2Ntot(phi_i, kwargs):
            """Convert input `phi_i` to `N_tot`."""
            ic = InflationStartIC(equations=self.equations,
                                  phi_i=phi_i,
                                  t_i=self.t_i,
                                  eta_i=self.eta_i,
                                  x_end=self.x_end,
                                  **self.ic_input_param)
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore',
                                        message="Inflation",
                                        category=UserWarning)
                sol = solve(ic, events=events, **kwargs)
            if np.isfinite(sol.N_tot):
                if self.verbose:
                    print("N_tot = %.15g" % sol.N_tot)
                return sol.N_tot - self.N_tot
            else:
                if (('Collapse' in sol.N_events and np.size(sol.N_events['Collapse']) > 0) or
                        ('Inflation_dir-1_term0' in sol.N_events and
                         sol.N_events['Inflation_dir-1_term0'] == sol.N[0]) or
                        ('Inflation_dir-1_term1' in sol.N_events and
                         sol.N_events['Inflation_dir-1_term1'] == sol.N[0])):
                    warnings.warn("foo")
                    return 0 - self.N_tot
                else:
                    print("sol = %s" % sol)
                    raise Exception("solve_ivp failed with message: %s" % sol.message)

        if isinstance(self.equations, InflationEquationsN):
            output = root_scalar(phii2Ntot, args=(ivp_kwargs,), bracket=self.phi_i_bracket,
                                 rtol=1e-6, xtol=1e-6)
        else:
            output = root_scalar(phii2Ntot, args=(ivp_kwargs,), bracket=self.phi_i_bracket)
        if self.verbose:
            print(output)
        phi_i_new = output.root
        super(ISIC_NiNt, self).__init__(equations=self.equations,
                                        phi_i=phi_i_new,
                                        t_i=self.t_i,
                                        eta_i=self.eta_i,
                                        x_end=self.x_end,
                                        **self.ic_input_param)
        super(ISIC_NiNt, self).__call__(y0=y0, **ivp_kwargs)
        return phi_i_new, output


# noinspection PyPep8Naming
class ISIC_NiNsOk(InflationStartIC):
    """Inflation start initial conditions given potential mass/Lambda, N_star, N_i, Omega_K0, h."""

    def __init__(self, equations, N_star, Omega_K0, h, phi_i_bracket, t_i=None, eta_i=None,
                 x_end=1e300, verbose=False, **kwargs):
        assert Omega_K0 != 0, "Curved universes only, here! Flat universes can set N_star freely."
        super(ISIC_NiNsOk, self).__init__(equations=equations,
                                          phi_i=phi_i_bracket[-1],
                                          t_i=t_i,
                                          eta_i=eta_i,
                                          x_end=x_end,
                                          **kwargs)
        self.N_star = N_star
        self.Omega_K0 = Omega_K0
        self.h = h
        self.phi_i_bracket = phi_i_bracket
        self.verbose = verbose

    def __call__(self, y0, **ivp_kwargs):
        """Set background equations of inflation optimizing for `N_star`."""
        events = [InflationEvent(self.equations, direction=+1, terminal=False),
                  InflationEvent(self.equations, direction=-1, terminal=True),
                  CollapseEvent(self.equations)]

        def phii2Nstar(phi_i, kwargs):
            """Convert input `phi_i` to `N_star`."""
            ic = InflationStartIC(equations=self.equations,
                                  phi_i=phi_i,
                                  t_i=self.t_i,
                                  eta_i=self.eta_i,
                                  x_end=self.x_end,
                                  **self.ic_input_param)
            sol = solve(ic, events=events, **kwargs)
            if sol.success and np.isfinite(sol.N_tot) and sol.N_tot > self.N_star:
                sol.derive_approx_power(Omega_K0=self.Omega_K0, h=self.h, kind='linear')
                if self.verbose:
                    print("N_tot = %.15g, \t N_star = %.15g" % (sol.N_tot, sol.N_star))
                return sol.N_star - self.N_star
            else:
                if np.size(sol.t_events['Collapse']) > 0 or sol.N_tot <= self.N_star:
                    return 0 - self.N_star
                else:
                    print("sol = %s" % sol)
                    raise Exception("solve_ivp failed with message: %s" % sol.message)

        if isinstance(self.equations, InflationEquationsN):
            output = root_scalar(phii2Nstar, args=(ivp_kwargs,), bracket=self.phi_i_bracket,
                                 rtol=1e-6, xtol=1e-6)
        else:
            output = root_scalar(phii2Nstar, args=(ivp_kwargs,), bracket=self.phi_i_bracket)
        phi_i_new = output.root
        super(ISIC_NiNsOk, self).__init__(equations=self.equations,
                                          phi_i=phi_i_new,
                                          t_i=self.t_i,
                                          eta_i=self.eta_i,
                                          x_end=self.x_end,
                                          **self.ic_input_param)
        super(ISIC_NiNsOk, self).__call__(y0=y0, **ivp_kwargs)
        return phi_i_new, output
