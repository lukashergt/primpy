#!/usr/bin/env python
""":mod:`primpy.initialconditions`: initial conditions for inflation."""
import warnings
import numpy as np
from scipy.optimize import root_scalar
from primpy.exceptionhandling import StepSizeError, PrimpyError, InflationStartError
from primpy.exceptionhandling import InflationWarning
from primpy.time.inflation import InflationEquationsT
from primpy.efolds.inflation import InflationEquationsN
from primpy.solver import solve
from primpy.events import InflationEvent, CollapseEvent, UntilNEvent
import primpy.bigbang as bb


# noinspection PyPep8Naming
class SlowRollIC(object):
    """Slow-roll initial conditions given `phi_i` and either of `N_i` or `Omega_Ki`.

    Class for setting up initial conditions during slow-roll inflation where
    the potential energy dominates over the kinetic energy.
    """

    def __init__(self, equations, phi_i, t_i=None, eta_i=None, x_end=1e300, **kwargs):
        self.equations = equations
        self.phi_i = phi_i
        self.t_i = t_i
        self.eta_i = eta_i
        self.x_end = x_end

        self.V_i = equations.potential.V(self.phi_i)
        self.dV_i = equations.potential.dV(self.phi_i)
        if 'N_i' in kwargs and 'Omega_Ki' not in kwargs:
            self.N_i = kwargs.pop('N_i')
            self.ic_input_param = {'N_i': self.N_i}
            self.aH2_i = self.V_i / 3 * np.exp(2 * self.N_i) - equations.K
            if self.aH2_i < 0:
                raise InflationStartError(
                    "V_i / 3 * exp(2 N_i) - 1 = %s < 0 but needs to be > 0. Increase either N_i "
                    "or phi_i." % self.aH2_i, geometry="closed")
            self.aH_i = np.sqrt(self.V_i / 3 * np.exp(2 * self.N_i) - equations.K)
            self.Omega_Ki = -equations.K / self.aH2_i
        elif 'Omega_Ki' in kwargs and 'N_i' not in kwargs:
            self.Omega_Ki = kwargs.pop('Omega_Ki')
            self.ic_input_param = {'Omega_Ki': self.Omega_Ki}
            if self.Omega_Ki >= 1:
                raise InflationStartError(
                    "Primordial curvature for open universes has to be Omega_Ki < 1, "
                    "but Omega_Ki = %g was requested." % self.Omega_Ki, geometry="open")
            self.N_i = np.log(3 * equations.K / self.V_i * (1 - 1 / self.Omega_Ki)) / 2
            self.aH2_i = -equations.K / self.Omega_Ki
            self.aH_i = np.sqrt(self.aH2_i)
        else:
            raise TypeError("Need to specify either N_i xor Omega_Ki.")
        self.H_i = self.aH_i * np.exp(-self.N_i)
        if isinstance(self.equations, InflationEquationsT):
            self.x_ini = self.t_i
            self.x_end = self.x_end
            self.dphidt_i = -self.dV_i / (3 * self.H_i)
        elif isinstance(self.equations, InflationEquationsN):
            self.x_ini = self.N_i
            self.x_end = self.x_end
            self.dphidN_i = -self.dV_i / (3 * self.H_i**2)

    def __call__(self, y0, **ivp_kwargs):
        """Set background equations of inflation for `_N`, `phi` and `dphi`."""
        if isinstance(self.equations, InflationEquationsT):
            y0[self.equations.idx['dphidt']] = self.dphidt_i
            y0[self.equations.idx['_N']] = self.N_i
        elif isinstance(self.equations, InflationEquationsN):
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


class InflationStartIC(object):
    """Inflation start initial conditions given `phi_i` and either of `N_i` or `Omega_Ki`.

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
            if self.V_i / 2 * np.exp(2 * self.N_i) - equations.K < 0:
                raise InflationStartError(
                    "V_i / 2 * exp(2 N_i) - 1 = %s < 0 but needs to be > 0. Increase either N_i "
                    "or phi_i." % (self.V_i / 2 * np.exp(2 * self.N_i) - 1), geometry="closed")
            self.aH_i = np.sqrt(self.V_i / 2 * np.exp(2 * self.N_i) - equations.K)
            self.Omega_Ki = -equations.K / self.aH_i**2
        elif 'Omega_Ki' in kwargs:
            assert 'N_i' not in kwargs, "Only either N_i or Omega_Ki should be specified. " \
                                        "The other will be inferred."
            self.Omega_Ki = kwargs.pop('Omega_Ki')
            self.ic_input_param = {'Omega_Ki': self.Omega_Ki}
            if self.Omega_Ki >= 1:
                raise InflationStartError(
                    "Primordial curvature for open universes has to be Omega_Ki < 1, "
                    "but Omega_Ki = %g was requested." % self.Omega_Ki, geometry="open")
            self.N_i = np.log(2 * equations.K / self.V_i * (1 - 1 / self.Omega_Ki)) / 2
            self.aH_i = np.sqrt(-equations.K / self.Omega_Ki)
        else:
            raise TypeError("Need to specify either N_i or Omega_Ki.")
        self.H_i = np.sqrt(self.V_i / 2 - equations.K * np.exp(-2 * self.N_i))

    def __call__(self, y0, **ivp_kwargs):
        """Set background equations of inflation for `_N`, `phi` and `dphi`."""
        if isinstance(self.equations, InflationEquationsT):
            self.x_ini = self.t_i
            self.x_end = self.x_end
            self.dphidt_i = -np.sqrt(self.V_i)
            y0[self.equations.idx['dphidt']] = self.dphidt_i
            y0[self.equations.idx['_N']] = self.N_i
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
class ISIC_Nt(InflationStartIC):
    """Inflation start initial conditions given `N_tot` and either of `N_i` or `Omega_Ki`."""

    def __init__(self, equations, N_tot, phi_i_bracket, t_i=None, eta_i=None,
                 x_end=1e300, verbose=False, **kwargs):
        super(ISIC_Nt, self).__init__(equations=equations,
                                      phi_i=phi_i_bracket[-1],
                                      t_i=t_i,
                                      eta_i=eta_i,
                                      x_end=x_end,
                                      **kwargs)
        self.N_tot = N_tot
        self.phi_i_bracket = phi_i_bracket
        self.warn_action = 'always' if verbose else 'ignore'
        self.vprint = print if verbose else lambda *a, **k: None
        self.equations.vwarn = warnings.warn if verbose else lambda *a, **k: None

    def __call__(self, y0, **ivp_kwargs):
        """Set background equations of inflation optimizing for `N_tot`."""

        def phii2Ntot(phi_i, kwargs):
            """Convert input `phi_i` to `N_tot`."""
            try:
                ic = InflationStartIC(equations=self.equations,
                                      phi_i=phi_i,
                                      t_i=self.t_i,
                                      eta_i=self.eta_i,
                                      x_end=self.x_end,
                                      **self.ic_input_param)
            except InflationStartError:
                return 0 - self.N_tot
            events = [InflationEvent(self.equations, direction=+1, terminal=False),
                      InflationEvent(self.equations, direction=-1, terminal=True),
                      UntilNEvent(self.equations, ic.N_i + self.N_tot + 10),
                      CollapseEvent(self.equations)]
            with warnings.catch_warnings():
                warnings.filterwarnings(action=self.warn_action, category=InflationWarning)
                sol = solve(ic, events=events, **kwargs)
            if np.isfinite(sol.N_tot):
                self.vprint("N_tot = %.15g for phi_i = %.15g" % (sol.N_tot, phi_i))
                return sol.N_tot - self.N_tot
            elif np.size(sol._N_events['UntilN']) > 0 or sol._N[-1] - ic.N_i >= self.N_tot:
                self.vprint("N_tot > %g for phi_i = %.15g" % (self.N_tot, phi_i))
                return sol._N[-1] - ic.N_i
            elif (np.size(sol._N_events['Collapse']) > 0 or
                  sol._N_events['Inflation_dir-1_term1'] == sol._N[0]):
                self.vprint("N_tot = %g for phi_i = %.15g" % (sol.N_tot, phi_i))
                return 0 - self.N_tot
            elif 'step size' in sol.message:
                raise StepSizeError(sol.message)
            else:
                self.vprint("sol = %s" % sol)
                raise PrimpyError("`solve_ivp` failed with message: %s" % sol.message)

        if isinstance(self.equations, InflationEquationsN):
            output = root_scalar(phii2Ntot, args=(ivp_kwargs,), bracket=self.phi_i_bracket,
                                 rtol=1e-6, xtol=1e-6)
        else:
            output = root_scalar(phii2Ntot, args=(ivp_kwargs,), bracket=self.phi_i_bracket)
        self.vprint(output)
        phi_i_new = output.root
        super(ISIC_Nt, self).__init__(equations=self.equations,
                                      phi_i=phi_i_new,
                                      t_i=self.t_i,
                                      eta_i=self.eta_i,
                                      x_end=self.x_end,
                                      **self.ic_input_param)
        super(ISIC_Nt, self).__call__(y0=y0, **ivp_kwargs)
        return phi_i_new, output


# noinspection PyPep8Naming
class ISIC_NsOk(InflationStartIC):
    """Inflation start initial conditions given `N_star`, `Omega_K0`, `h`.

    Additionally either `N_i` or `Omega_Ki` need to be specified.

    """

    def __init__(self, equations, N_star, Omega_K0, h, phi_i_bracket, t_i=None, eta_i=None,
                 x_end=1e300, verbose=False, **kwargs):
        assert Omega_K0 != 0, "Curved universes only, here! Flat universes can set N_star freely."
        super(ISIC_NsOk, self).__init__(equations=equations,
                                        phi_i=phi_i_bracket[-1],
                                        t_i=t_i,
                                        eta_i=eta_i,
                                        x_end=x_end,
                                        **kwargs)
        self.N_star = N_star
        self.Omega_K0 = Omega_K0
        self.h = h
        self.phi_i_bracket = phi_i_bracket
        self.warn_action = 'always' if verbose else 'ignore'
        self.vprint = print if verbose else lambda *a, **k: None
        self.vwarn = warnings.warn if verbose else lambda *a, **k: None
        self.equations.vwarn = self.vwarn
        self.a0 = bb.get_a0(h=h, Omega_K0=Omega_K0, units='planck')
        self.N0 = np.log(self.a0)

    def __call__(self, y0, **ivp_kwargs):
        """Set background equations of inflation optimizing for `N_star`."""

        def phii2Nstar(phi_i, kwargs):
            """Convert input `phi_i` to `N_star`."""
            ic = InflationStartIC(equations=self.equations,
                                  phi_i=phi_i,
                                  t_i=self.t_i,
                                  eta_i=self.eta_i,
                                  x_end=self.x_end,
                                  **self.ic_input_param)
            events = [InflationEvent(self.equations, direction=+1, terminal=False),
                      InflationEvent(self.equations, direction=-1, terminal=True),
                      UntilNEvent(self.equations, self.N0),
                      CollapseEvent(self.equations)]
            with warnings.catch_warnings():
                warnings.filterwarnings(action=self.warn_action, category=InflationWarning)
                sol = solve(ic, events=events, **kwargs)
            if np.isfinite(sol.N_tot) and sol.N_tot > self.N_star:
                sol.calibrate_scale_factor(Omega_K0=self.Omega_K0, h=self.h)
                sol.derive_approx_power()
                self.vprint("N_tot = %.15g, N_star = %.15g for phi_i = %.15g"
                            % (sol.N_tot, sol.N_star, phi_i))
                return sol.N_star - self.N_star
            elif np.size(sol._N_events['UntilN']) > 0 or sol._N[-1] >= self.N0:
                self.vprint("N_tot > %g for phi_i = %.15g" % (self.N0, phi_i))
                return sol._N[-1]
            elif (np.size(sol._N_events['Collapse']) > 0 or sol.N_tot <= self.N_star or
                  sol._N_events['Inflation_dir-1_term1'] == sol._N[0]):
                self.vprint("N_tot = %g for phi_i = %.15g" % (sol.N_tot, phi_i))
                if sol.N_tot <= self.N_star:
                    self.vwarn(InflationWarning("Insufficient inflation: N_tot = %g < %g = N_star"
                                                % (sol.N_tot, self.N_star)))
                elif sol._N_events['Inflation_dir-1_term1'] == sol._N[0]:
                    self.vwarn(InflationWarning("Universe has ended early: _N[0]=%g, _N_events=%s"
                                                % (sol._N[0], sol._N_events)))
                return 0 - self.N_star
            elif 'step size' in sol.message:
                raise StepSizeError(sol.message)
            else:
                self.vprint("sol = %s" % sol)
                raise PrimpyError("`solve_ivp` failed with message: %s" % sol.message)

        if isinstance(self.equations, InflationEquationsN):
            output = root_scalar(phii2Nstar, args=(ivp_kwargs,), bracket=self.phi_i_bracket,
                                 rtol=1e-6, xtol=1e-6)
        else:
            output = root_scalar(phii2Nstar, args=(ivp_kwargs,), bracket=self.phi_i_bracket)
        self.vprint(output)
        phi_i_new = output.root
        super(ISIC_NsOk, self).__init__(equations=self.equations,
                                        phi_i=phi_i_new,
                                        t_i=self.t_i,
                                        eta_i=self.eta_i,
                                        x_end=self.x_end,
                                        **self.ic_input_param)
        super(ISIC_NsOk, self).__call__(y0=y0, **ivp_kwargs)
        self.vprint("finished ic with phi_i_new=%f" % phi_i_new)
        return phi_i_new, output
