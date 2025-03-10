"""Inflationary potentials."""
from abc import ABC, abstractmethod
from functools import cached_property
from warnings import warn
import numpy as np
from scipy.special import lambertw
from scipy.interpolate import interp1d
from primpy.units import pi
from primpy.exceptionhandling import PrimpyError, PrimpyWarning


class InflationaryPotential(ABC):
    """Base class for inflaton potential and derivatives."""

    @abstractmethod
    def __init__(self, **pot_kwargs):
        self.Lambda = pot_kwargs.pop('Lambda', 1)
        if 'A_s' in pot_kwargs:
            if self.Lambda != 1:
                warn(PrimpyWarning("When specifying `A_s` alongside `Lambda`, the latter input "
                                   "will be ignored and instead inferred from `A_s` using the "
                                   "slow-roll approximation."))
            if 'N_star' not in pot_kwargs and 'phi_star' not in pot_kwargs:
                raise PrimpyError("When specifying `A_s`, need to also specify either `N_star` or "
                                  "`phi_star`.")
            A_s = pot_kwargs.pop('A_s')
            N_star = pot_kwargs.pop('N_star', None)
            phi_star = pot_kwargs.pop('phi_star', None)
            self.Lambda, _, _ = self.sr_As2Lambda(A_s=A_s, N_star=N_star, phi_star=phi_star)
        for key in pot_kwargs:
            raise Exception("%s does not accept kwarg %s" % (self.name, key))

    @property
    @abstractmethod
    def tag(self):
        """3 letter tag identifying the type of inflationary potential."""

    @property
    @abstractmethod
    def name(self):
        """Name of the inflationary potential."""

    @property
    @abstractmethod
    def tex(self):
        """Tex string useful for labelling the inflationary potential."""

    @property
    @abstractmethod
    def perturbation_ic(self):
        """Set of well scaling initial conditions for perturbation module."""

    @abstractmethod
    def V(self, phi):
        """Inflationary potential `V(phi)`.

        Parameters
        ----------
        phi : float or np.ndarray
            Inflaton field `phi`.

        Returns
        -------
        V : float or np.ndarray
            Inflationary potential `V(phi)`.

        """

    @abstractmethod
    def dV(self, phi):
        """First derivative `V'(phi)` with respect to inflaton `phi`.

        Parameters
        ----------
        phi : float or np.ndarray
            Inflaton field `phi`.

        Returns
        -------
        dV : float or np.ndarray
            1st derivative of inflationary potential: `V'(phi)`.

        """

    @abstractmethod
    def d2V(self, phi):
        """Second derivative `V''(phi)` with respect to inflaton `phi`.

        Parameters
        ----------
        phi : float or np.ndarray
            Inflaton field `phi`.

        Returns
        -------
        d2V : float or np.ndarray
            2nd derivative of inflationary potential: `V''(phi)`.

        """

    @abstractmethod
    def d3V(self, phi):
        """Third derivative `V'''(phi)` with respect to inflaton `phi`.

        Parameters
        ----------
        phi : float or np.ndarray
            Inflaton field `phi`.

        Returns
        -------
        d3V : float or np.ndarray
            3rd derivative of inflationary potential: `V'''(phi)`.

        """

    @abstractmethod
    def d4V(self, phi):
        """Fourth derivative `V''''(phi)` with respect to inflaton `phi`.

        Parameters
        ----------
        phi : float or np.ndarray
            Inflaton field `phi`.

        Returns
        -------
        d4V : float or np.ndarray
            4th derivative of inflationary potential: `V''''(phi)`.

        """

    @abstractmethod
    def inv_V(self, V):
        """Inverse function `phi(V)` with respect to potential `V`.

        Parameters
        ----------
        V : float or np.ndarray
            Inflationary potential `V`.

        Returns
        -------
        phi : float or np.ndarray
            Inflaton field `phi`.

        """

    # TODO:
    # @abstractmethod
    # def sr_n_s(self):
    #     """Slow-roll approximation for the spectral index."""

    # TODO:
    # @abstractmethod
    # def sr_n_s(self):
    #     """Slow-roll approximation for the tensor-to-scalar ratio."""

    @abstractmethod
    def get_epsilon_1V(self, phi):
        """Get 1st potential slow-roll parameter."""

    @abstractmethod
    def get_epsilon_2V(self, phi):
        """Get 2nd potential slow-roll parameter."""

    @abstractmethod
    def get_epsilon_3V(self, phi):
        """Get 3rd potential slow-roll parameter."""

    @abstractmethod
    def get_epsilon_4V(self, phi):
        """Get 4th potential slow-roll parameter."""

    def get_epsilon_1(self, phi):
        """Approximation of 1st Hubble flow parameter with potential slow-roll parameters."""
        e1V = self.get_epsilon_1V(phi=phi)
        e2V = self.get_epsilon_2V(phi=phi)
        e3V = self.get_epsilon_3V(phi=phi)
        return e1V - e1V * e2V / 3 - e1V**2 * e2V / 9 + 5/36 * e1V * e2V**2 + e1V * e2V * e3V / 9

    def get_epsilon_2(self, phi):
        """Approximation of 2nd Hubble flow parameter with potential slow-roll parameters."""
        e1V = self.get_epsilon_1V(phi=phi)
        e2V = self.get_epsilon_2V(phi=phi)
        e3V = self.get_epsilon_3V(phi=phi)
        e4V = self.get_epsilon_4V(phi=phi)
        return (e2V - 1/6*e2V**2 - 1/3*e2V*e3V - 1/6*e1V*e2V**2 + 1/18*e2V**3 - 1/9*e1V*e2V*e3V
                + 5/18*e2V**2*e3V + 1/9*e2V*e3V**2 + 1/9*e2V*e3V*e4V)

    def get_epsilon_3(self, phi):
        """Approximation of 3rd Hubble flow parameter with potential slow-roll parameters."""
        e1V = self.get_epsilon_1V(phi=phi)
        e2V = self.get_epsilon_2V(phi=phi)
        e3V = self.get_epsilon_3V(phi=phi)
        e4V = self.get_epsilon_4V(phi=phi)
        e5V = 0
        return (e3V - 1/3*e2V*e3V - 1/3*e3V*e4V - 1/6*e1V*e2V**2 - 1/3*e1V*e2V*e3V
                + 1/6*e2V**2*e3V + 5/18*e2V*e3V**2 - 1/9*e1V*e3V*e4V + 5/18*e2V*e3V*e4V
                + 1/9*e3V**2*e4V + 1/9*e3V*e4V**2 + 1/9*e3V*e4V*e5V)

    def get_epsilon_4(self, phi):
        """Approximation of 4th Hubble flow parameter with potential slow-roll parameters."""
        e2V = self.get_epsilon_2V(phi=phi)
        e3V = self.get_epsilon_3V(phi=phi)
        e4V = self.get_epsilon_4V(phi=phi)
        e5V = 0
        return e4V - 1/3*e2V*e3V - 1/6*e2V*e4V - 1/3*e4V*e5V

    @abstractmethod
    @cached_property
    def phi_end(self):
        """Inflaton field value at the end of inflation.

        Value inferred from `epsilon_1V = 1` assuming a positive `phi_end`.
        """

    @abstractmethod
    def sr_phi2N(self, phi):
        """Convert from inflaton field `phi` to e-folds `N` assuming slow-roll approximation."""

    @abstractmethod
    def sr_N2phi(self, N):
        """Convert from inflaton field `phi` to e-folds `N` assuming slow-roll approximation."""

    def sr_As2Lambda(self, A_s, phi_star, N_star, **pot_kwargs):
        """Get potential amplitude `Lambda` from PPS amplitude `A_s` assuming slow-roll."""
        if N_star is None:
            N_star = self.sr_phi2N(phi_star)
        elif phi_star is None:
            phi_star = self.sr_N2phi(N_star)
        else:
            raise Exception("Need to specify either N_star or phi_star. "
                            "The respective other should be None.")

        e1 = self.get_epsilon_1V(phi=phi_star)
        H2 = self.V(phi=phi_star)/3  # slow-roll approximation at leading order
        A_s0 = H2 / (8*pi**2*e1)
        Lambda = self.Lambda * (A_s / A_s0)**(1/4)
        return Lambda, phi_star, N_star


class MonomialPotential(InflationaryPotential):
    """Monomial potential: `V(phi) = Lambda**4 * phi**p`."""

    tag = 'mnp'
    name = 'MonomialPotential'
    tex = r'$\phi^p$'
    perturbation_ic = (1, 0, 0, 1)

    def __init__(self, **pot_kwargs):
        self.p = pot_kwargs.pop('p')
        super().__init__(**pot_kwargs)

    def V(self, phi):  # noqa: D102
        return self.Lambda**4 * np.abs(phi)**self.p

    def dV(self, phi):  # noqa: D102
        return self.Lambda**4 * self.p * np.abs(phi)**(self.p - 1) * np.sign(phi)

    def d2V(self, phi):  # noqa: D102
        return self.Lambda**4 * self.p * (self.p - 1) * np.abs(phi)**self.p / phi**2

    def d3V(self, phi):  # noqa: D102
        p = self.p
        return self.Lambda**4 * p * (p**2 - 3*p + 2) * np.abs(phi)**(p-3) * np.sign(phi)

    def d4V(self, phi):  # noqa: D102
        p = self.p
        return self.Lambda**4 * p * (p**3 - 6*p**2 + 11*p - 6) * np.abs(phi)**(p-4)

    def inv_V(self, V):  # noqa: D102
        return (V / self.Lambda**4)**(1/self.p)

    def get_epsilon_1V(self, phi):  # noqa: D102
        return self.p**2 / (2 * phi**2)

    def get_epsilon_2V(self, phi):  # noqa: D102
        return 2 * self.p / phi**2

    def get_epsilon_3V(self, phi):  # noqa: D102
        return 2 * self.p / phi**2

    def get_epsilon_4V(self, phi):  # noqa: D102
        return 2 * self.p / phi**2

    @cached_property
    def phi_end(self):  # noqa: D102
        return np.sqrt(2) * self.p / 2

    def sr_phi2N(self, phi):  # noqa: D102
        return -self.p / 4 + phi**2 / (2 * self.p)

    def sr_N2phi(self, N):  # noqa: D102
        return np.sqrt(2 * self.p) * np.sqrt(4 * N + self.p) / 2

    @staticmethod
    def sr_Nstar2ns(N_star, **pot_kwargs):
        """Slow-roll approximation for inferring `n_s` from `N_star`."""
        p = pot_kwargs.pop('p')
        return 1 - p / (2 * N_star) - 1 / N_star

    @staticmethod
    def sr_ns2Nstar(n_s, **pot_kwargs):
        """Slow-roll approximation for inferring `N_star` from `n_s`."""
        p = pot_kwargs.pop('p')
        return (2 + p) / (2 * (1 - n_s))

    @staticmethod
    def sr_Nstar2r(N_star, **pot_kwargs):
        """Slow-roll approximation for inferring `r` from `N_star`."""
        p = pot_kwargs.pop('p')
        return 16 * p / (4 * N_star + p)

    @staticmethod
    def sr_r2Nstar(r, **pot_kwargs):
        """Slow-roll approximation for inferring `N_star` from `r`."""
        p = pot_kwargs.pop('p')
        return p * (16 - r) / (4 * r)


class LinearPotential(MonomialPotential):
    """Linear potential: `V(phi) = Lambda**4 * phi`."""

    tag = 'mn1'
    name = 'LinearPotential'
    tex = r'$\phi^1$'
    perturbation_ic = (1, 0, 0, 1)

    def __init__(self, **pot_kwargs):
        super().__init__(p=1, **pot_kwargs)

    @staticmethod
    def sr_Nstar2ns(N_star, **pot_params):  # noqa: D102
        return MonomialPotential.sr_Nstar2ns(N_star=N_star, p=1)

    @staticmethod
    def sr_ns2Nstar(n_s, **pot_params):  # noqa: D102
        return MonomialPotential.sr_ns2Nstar(n_s=n_s, p=1)

    @staticmethod
    def sr_Nstar2r(N_star, **pot_params):  # noqa: D102
        return MonomialPotential.sr_Nstar2r(N_star=N_star, p=1)

    @staticmethod
    def sr_r2Nstar(r, **pot_params):  # noqa: D102
        return MonomialPotential.sr_r2Nstar(r=r, p=1)


class QuadraticPotential(MonomialPotential):
    """Quadratic potential: `V(phi) = Lambda**4 * phi**2`."""

    tag = 'mn2'
    name = 'QuadraticPotential'
    tex = r'$\phi^2$'
    perturbation_ic = (1, 0, 0, 1)

    def __init__(self, **pot_kwargs):
        if 'mass' in pot_kwargs:
            raise ValueError("'mass' was dropped use 'Lambda' instead: Lambda**4=mass**2")
        super().__init__(p=2, **pot_kwargs)

    def V(self, phi):  # noqa: D102
        return self.Lambda**4 * phi**2

    def dV(self, phi):  # noqa: D102
        return 2 * self.Lambda**4 * phi

    def d2V(self, phi):  # noqa: D102
        return 2 * self.Lambda**4

    def d3V(self, phi):  # noqa: D102
        return 0

    def d4V(self, phi):  # noqa: D102
        return 0

    def inv_V(self, V):  # noqa: D102
        return np.sqrt(V) / self.Lambda**2

    @staticmethod
    def sr_Nstar2ns(N_star, **pot_kwargs):  # noqa: D102
        return MonomialPotential.sr_Nstar2ns(N_star=N_star, p=2)

    @staticmethod
    def sr_ns2Nstar(n_s, **pot_kwargs):  # noqa: D102
        return MonomialPotential.sr_ns2Nstar(n_s=n_s, p=2)

    @staticmethod
    def sr_Nstar2r(N_star, **pot_kwargs):  # noqa: D102
        return MonomialPotential.sr_Nstar2r(N_star=N_star, p=2)

    @staticmethod
    def sr_r2Nstar(r, **pot_kwargs):  # noqa: D102
        return MonomialPotential.sr_r2Nstar(r=r, p=2)


class CubicPotential(MonomialPotential):
    """Linear potential: `V(phi) = Lambda**4 * phi`."""

    tag = 'mn3'
    name = 'CubicPotential'
    tex = r'$\phi^3$'
    perturbation_ic = (1, 0, 0, 1)

    def __init__(self, **pot_kwargs):
        super().__init__(p=3, **pot_kwargs)

    @staticmethod
    def sr_Nstar2ns(N_star, **pot_kwargs):  # noqa: D102
        return MonomialPotential.sr_Nstar2ns(N_star=N_star, p=3)

    @staticmethod
    def sr_ns2Nstar(n_s, **pot_kwargs):  # noqa: D102
        return MonomialPotential.sr_ns2Nstar(n_s=n_s, p=3)

    @staticmethod
    def sr_Nstar2r(N_star, **pot_kwargs):  # noqa: D102
        return MonomialPotential.sr_Nstar2r(N_star=N_star, p=3)

    @staticmethod
    def sr_r2Nstar(r, **pot_kwargs):  # noqa: D102
        return MonomialPotential.sr_r2Nstar(r=r, p=3)


class QuarticPotential(MonomialPotential):
    """Linear potential: `V(phi) = Lambda**4 * phi`."""

    tag = 'mn4'
    name = 'QuarticPotential'
    tex = r'$\phi^4$'
    perturbation_ic = (1, 0, 0, 1)

    def __init__(self, **pot_kwargs):
        super().__init__(p=4, **pot_kwargs)

    @staticmethod
    def sr_Nstar2ns(N_star, **pot_kwargs):  # noqa: D102
        return MonomialPotential.sr_Nstar2ns(N_star=N_star, p=4)

    @staticmethod
    def sr_ns2Nstar(n_s, **pot_kwargs):  # noqa: D102
        return MonomialPotential.sr_ns2Nstar(n_s=n_s, p=4)

    @staticmethod
    def sr_Nstar2r(N_star, **pot_kwargs):  # noqa: D102
        return MonomialPotential.sr_Nstar2r(N_star=N_star, p=4)

    @staticmethod
    def sr_r2Nstar(r, **pot_kwargs):  # noqa: D102
        return MonomialPotential.sr_r2Nstar(r=r, p=4)


class StarobinskyPotential(InflationaryPotential):
    """Starobinsky potential: `V(phi) = Lambda**4 * (1 - exp(-sqrt(2/3) * phi))**2`."""

    tag = 'stb'
    name = 'StarobinskyPotential'
    tex = r'Starobinsky'
    gamma = np.sqrt(2 / 3)
    perturbation_ic = (1, 0, 0, 1)

    def __init__(self, **pot_kwargs):
        super().__init__(**pot_kwargs)

    def V(self, phi):  # noqa: D102
        return self.Lambda**4 * (1 - np.exp(-StarobinskyPotential.gamma * phi))**2

    def dV(self, phi):  # noqa: D102
        gamma = StarobinskyPotential.gamma
        return self.Lambda**4 * 2 * gamma * np.exp(-2 * gamma * phi) * (np.exp(gamma * phi) - 1)

    def d2V(self, phi):  # noqa: D102
        gamma = StarobinskyPotential.gamma
        return self.Lambda**4 * 2 * gamma**2 * np.exp(-2 * gamma * phi) * (2 - np.exp(gamma * phi))

    def d3V(self, phi):  # noqa: D102
        gamma = StarobinskyPotential.gamma
        return self.Lambda**4 * 2 * gamma**3 * np.exp(-2 * gamma * phi) * (np.exp(gamma * phi) - 4)

    def d4V(self, phi):  # noqa: D102
        gamma = StarobinskyPotential.gamma
        return 2 * self.Lambda**4 * gamma**4 * (8 - np.exp(gamma * phi)) * np.exp(-2 * gamma * phi)

    def inv_V(self, V):  # noqa: D102
        return -np.log(1 - np.sqrt(V) / self.Lambda**2) / StarobinskyPotential.gamma

    def get_epsilon_1V(self, phi):  # noqa: D102
        gamma = StarobinskyPotential.gamma
        return 2 * gamma**2 / (1 - np.exp(gamma * phi))**2

    def get_epsilon_2V(self, phi):  # noqa: D102
        gamma = StarobinskyPotential.gamma
        return gamma**2 / np.sinh(gamma * phi / 2)**2

    def get_epsilon_3V(self, phi):  # noqa: D102
        gamma = StarobinskyPotential.gamma
        return 2 * gamma**2 * (np.exp(gamma * phi) + 1) / (np.exp(gamma * phi) - 1)**2

    def get_epsilon_4V(self, phi):  # noqa: D102
        gamma = StarobinskyPotential.gamma
        return gamma**2 * (np.exp(gamma*phi)+3) / (2*(np.exp(gamma*phi)+1)*np.sinh(gamma*phi/2)**2)

    @cached_property
    def phi_end(self):  # noqa: D102
        gamma = StarobinskyPotential.gamma
        return np.log(np.sqrt(2) * gamma + 1) / gamma

    def sr_phi2N(self, phi):  # noqa: D102
        g = StarobinskyPotential.gamma
        phi_end = self.phi_end
        return (-phi + phi_end) / (2*g) + (np.exp(g * phi) - np.exp(g * phi_end)) / (2*g**2)

    def sr_N2phi(self, N):  # noqa: D102
        g = StarobinskyPotential.gamma
        phi_end = self.phi_end
        return (-2*N*g + phi_end - np.exp(g*phi_end)/g -
                lambertw(-np.exp(-2*N*g**2)*np.exp(g*phi_end)*np.exp(-np.exp(g*phi_end)), -1) / g)

    @staticmethod
    def sr_Nstar2ns(N_star):
        """Slow-roll approximation for inferring `n_s` from `N_star`."""
        gamma = StarobinskyPotential.gamma
        num = 2 * N_star * gamma**2 + np.sqrt(2) * gamma + 2
        den = N_star * gamma * (N_star * gamma + np.sqrt(2))
        return 1 - num / den

    @staticmethod
    def sr_ns2Nstar(n_s):
        """Slow-roll approximation for inferring `N_star` from `n_s`."""
        gamma = StarobinskyPotential.gamma
        num = 2*gamma - np.sqrt(2) * (1-n_s) + np.sqrt(2*(1-n_s)**2 + 8*(1-n_s) + 4*gamma**2)
        den = 2 * gamma * (1-n_s)
        return num / den

    @staticmethod
    def sr_Nstar2r(N_star):
        """Slow-roll approximation for inferring `r` from `N_star`."""
        gamma = StarobinskyPotential.gamma
        return 32 / (2*N_star*gamma + np.sqrt(2))**2

    @staticmethod
    def sr_r2Nstar(r):
        """Slow-roll approximation for inferring `N_star` from `r`."""
        gamma = StarobinskyPotential.gamma
        return np.sqrt(2) * (4 - np.sqrt(r)) / (2 * gamma * np.sqrt(r))

    @staticmethod
    def phi2efolds(phi):
        """Get e-folds `N` from inflaton `phi`.

        Find the number of e-folds `N` till end of inflation from inflaton `phi`
        using the slow-roll approximation.

        Parameters
        ----------
        phi : float or np.ndarray
            Inflaton field `phi`.

        Returns
        -------
        N : float or np.ndarray
            Number of e-folds `N` until end of inflation.

        """
        gamma = StarobinskyPotential.gamma
        phi_end = np.log(1 + np.sqrt(2) * gamma) / gamma  # =~ 0.9402
        return (np.exp(gamma * phi) - np.exp(gamma * phi_end)
                - gamma * (phi - phi_end)) / (2 * gamma**2)


class NaturalPotential(InflationaryPotential):
    """Natural inflation potential: `V(phi) = Lambda**4 * (1 - cos(pi*phi/phi0))`.

    Natural inflation with phi0 = pi * f where f is the standard parameter
    used in definitions of natural inflation.
    Here we use phi0 the position of the maximum and we have a minus in our
    definition such that the minimum is at zero instead of the maximum.
    """

    tag = 'nat'
    name = 'NaturalPotential'
    tex = r'Natural'
    perturbation_ic = (1, 0, 0, 1)

    def __init__(self, **pot_kwargs):
        self.phi0 = pot_kwargs.pop('phi0')
        super().__init__(**pot_kwargs)

    def V(self, phi):  # noqa: D102
        return self.Lambda**4 / 2 * (1 - np.cos(pi * phi / self.phi0))

    def dV(self, phi):  # noqa: D102
        return self.Lambda**4 / 2 * np.sin(pi * phi / self.phi0) * pi / self.phi0

    def d2V(self, phi):  # noqa: D102
        return self.Lambda**4 / 2 * np.cos(pi * phi / self.phi0) * (pi / self.phi0)**2

    def d3V(self, phi):  # noqa: D102
        return -self.Lambda**4 / 2 * np.sin(pi * phi / self.phi0) * (pi / self.phi0)**3

    def d4V(self, phi):  # noqa: D102
        return -self.Lambda**4 / 2 * np.cos(pi * phi / self.phi0) * (pi / self.phi0)**4

    def inv_V(self, V):  # noqa: D102
        return np.arccos(1 - 2 * V / self.Lambda**4) * self.phi0 / pi

    def get_epsilon_1V(self, phi):  # noqa: D102
        phi0 = self.phi0
        return pi**2 * np.sin(pi*phi/phi0)**2 / (2 * phi0**2 * (np.cos(pi*phi/phi0) - 1)**2)

    def get_epsilon_2V(self, phi):  # noqa: D102
        phi0 = self.phi0
        return -2 * pi**2 / (phi0**2 * (np.cos(pi*phi/phi0) - 1))

    def get_epsilon_3V(self, phi):  # noqa: D102
        phi0 = self.phi0
        return pi**2 * np.sin(pi*phi/phi0)**2 / (phi0**2 * (np.cos(pi*phi/phi0) - 1)**2)

    def get_epsilon_4V(self, phi):  # noqa: D102
        phi0 = self.phi0
        return -2 * pi**2 / (phi0**2 * (np.cos(pi*phi/phi0) - 1))

    @cached_property
    def phi_end(self):  # noqa: D102
        return 2 * self.phi0 * np.arctan(np.sqrt(2) * pi / (2 * self.phi0)) / pi

    def sr_phi2N(self, phi):  # noqa: D102
        phi0 = self.phi0
        phi_end = self.phi_end
        return phi0**2/pi**2 * (+ np.log(np.tan(pi*phi/(2*phi0))**2 + 1)
                                - np.log(np.tan(pi*phi_end/(2*phi0))**2 + 1))

    def sr_N2phi(self, N):  # noqa: D102
        phi0 = self.phi0
        phi_end = self.phi_end
        return 2*phi0/pi * np.arctan(np.sqrt((2*np.exp(pi**2*N/phi0**2)-np.cos(pi*phi_end/phi0)-1)
                                             / (np.cos(pi*phi_end/phi0) + 1)))

    @staticmethod
    def sr_Nstar2ns(N_star, **pot_kwargs):
        """Slow-roll approximation for the spectral index `n_s`."""
        phi0 = pot_kwargs.pop('phi0')
        f = phi0 / pi
        num = (2 * f**2 + (2 * f**2 + 1) * np.exp(N_star / f**2))
        den = (f**2 * (2 * f**2 + 1) * (np.exp(N_star / f**2) - 1))
        return 1 - num / den

    @staticmethod
    def sr_ns2Nstar(n_s, **pot_kwargs):
        """Slow-roll approximation for inferring `N_star` from `n_s`."""
        phi0 = pot_kwargs.pop('phi0')
        if phi0 < pi / np.sqrt(1-n_s):
            raise PrimpyError(f"Need phi0 > pi / np.sqrt(1-n_s) for Natural inflation, but "
                              f"phi0={phi0}, n_s={n_s}, and pi/sqrt(1-ns)={pi/np.sqrt(1-n_s)}.")
        f = phi0 / pi
        return f**2 * np.log(f**2 * (2*f**2*(1-n_s)+(1-n_s)+2) / ((2*f**2+1) * (f**2*(1-n_s)-1)))

    @staticmethod
    def sr_Nstar2r(N_star, **pot_kwargs):
        """Slow-roll approximation for the tensor-to-scalar ratio `r`."""
        phi0 = pot_kwargs.pop('phi0')
        f = phi0 / pi
        return 16 / (-2 * f**2 + (2 * f**2 + 1) * np.exp(N_star / f**2))

    @staticmethod
    def sr_r2Nstar(r, **pot_kwargs):
        """Slow-roll approximation for inferring `N_star` from `r`."""
        phi0 = pot_kwargs.pop('phi0')
        f = phi0 / pi
        return f**2 * np.log((2 * f**2 * r + 16) / (r * (2 * f**2 + 1)))

    @staticmethod
    def phi2efolds(phi, phi0):
        """Get e-folds `N` from inflaton `phi`.

        Find the number of e-folds `N` till end of inflation from inflaton `phi`
        using the slow-roll approximation.

        Parameters
        ----------
        phi : float or np.ndarray
            Inflaton field `phi`.
        phi0 : float
            Inflaton distance between local maximum and minima.

        Returns
        -------
        N : float or np.ndarray
            Number of e-folds `N` until end of inflation.

        """
        assert np.all(phi < phi0)
        f = phi0 / pi
        return -f**2 * (np.log(1 + 1 / (2 * f**2)) + 2 * np.log(np.cos(phi / (2 * f))))


class DoubleWellPotential(InflationaryPotential):
    """Double-Well potential: `V(phi) = Lambda**4 * (1 - (phi/phi0)**p)**2`.

    Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
    """

    tag = 'dwp'
    name = 'DoubleWellPotential'
    tex = r'Double-Well (p)'
    perturbation_ic = (1, 0, 0, 1)

    def __init__(self, **pot_kwargs):
        self.phi0 = pot_kwargs.pop('phi0')
        self.p = pot_kwargs.pop('p')
        super().__init__(**pot_kwargs)

    def V(self, phi):  # noqa: D102
        phi_0_phi = (self.phi0 - phi) / self.phi0
        return self.Lambda**4 * (1 - phi_0_phi**self.p)**2

    def dV(self, phi):  # noqa: D102
        p = self.p
        phi0 = self.phi0
        phi_0_phi = (phi0 - phi) / phi0
        pre = self.Lambda**4 * 2 * p * phi_0_phi**(p-1) / phi0
        return pre * (1 - phi_0_phi**p)

    def d2V(self, phi):  # noqa: D102
        p = self.p
        phi0 = self.phi0
        phi_0_phi = (phi0 - phi) / phi0
        pre = self.Lambda**4 * 2 * p * phi_0_phi**(p-2) / phi0**2
        return pre * (-p + phi_0_phi**p * (2*p-1) + 1)

    def d3V(self, phi):  # noqa: D102
        p = self.p
        phi0 = self.phi0
        phi_0_phi = (phi0 - phi) / phi0
        pre = self.Lambda**4 * 2 * p * phi_0_phi**(p-3) / phi0**3
        return pre * (p**2 - 3*p + phi_0_phi**p * (-4*p**2 + 6*p - 2) + 2)

    def d4V(self, phi):  # noqa: D102
        p = self.p
        phi0 = self.phi0
        phi_0_phi = (phi0 - phi) / phi0
        pre = self.Lambda**4 * 2 * p * phi_0_phi**(p-4) / phi0**4
        return pre * (-p**3 + 6*p**2 - 11*p + phi_0_phi**p * (8*p**3-24*p**2+22*p-6) + 6)

    def inv_V(self, V):  # noqa: D102
        return self.phi0 - self.phi0 * (1 - np.sqrt(V) / self.Lambda**2)**(1/self.p)

    def get_epsilon_1V(self, phi):  # noqa: D102
        p = self.p
        phi0 = self.phi0
        phi_0_phi = (phi0 - phi) / phi0
        return 2 * p**2 * phi_0_phi**(2*p-2) / (phi0**2 * (phi_0_phi**p - 1)**2)

    def get_epsilon_2V(self, phi):  # noqa: D102
        p = self.p
        phi0 = self.phi0
        phi_0_phi = (phi0 - phi) / phi0
        return (4 * p * phi_0_phi**(p - 2) * (p + phi_0_phi**p - 1)
                / (phi0**2 * (phi_0_phi**p - 1)**2))

    def get_epsilon_3V(self, phi):  # noqa: D102
        p = self.p
        phi0 = self.phi0
        phi_0_phi = (phi0 - phi) / phi0
        return (2*p*phi_0_phi**(p-2) * (p**2-3*p+2*phi_0_phi**(2*p)+phi_0_phi**p*(p**2+3*p-4)+2)
                / (phi0**2 * (phi_0_phi**p - 1)**2 * (p + phi_0_phi**p - 1)))

    def get_epsilon_4V(self, phi):  # noqa: D102
        p = self.p
        phi0 = self.phi0
        phi_0_phi = (phi0 - phi) / phi0
        num = 2 * p * phi_0_phi**(p-2) * (p**4 - 6*p**3 + 13*p**2 - 12*p
                                          + 4*phi_0_phi**(4*p)
                                          + phi_0_phi**(3*p) * (p**3 + 3*p**2 + 12*p - 16)
                                          + phi_0_phi**(2*p) * (5*p**3 + 7*p**2 - 36*p + 24)
                                          + phi_0_phi**p * (3*p**4 - 23*p**2 + 36*p - 16) + 4)
        den = phi0**2 * (p**3 - 4 * p**2 + 5 * p
                         + 2 * phi_0_phi**(5*p)
                         + phi_0_phi**(4*p) * (p**2 + 5*p - 10)
                         + phi_0_phi**(3*p) * (p**3 + p**2 - 20*p + 20)
                         + phi_0_phi**(2*p) * (-p**3 - 9*p**2 + 30*p - 20)
                         + phi_0_phi**p * (-p**3 + 11*p**2 - 20*p + 10) - 2)
        return num / den

    @cached_property
    def phi_end(self):  # noqa: D102
        phi0 = self.phi0
        phis = np.linspace(0, phi0, 10001)[1:-1]
        e1V = self.get_epsilon_1V(phi=phis)
        loge2phi = interp1d(np.log10(e1V), phis)
        phi_end = loge2phi(0).item()
        return phi_end

    def sr_phi2N(self, phi):  # noqa: D102
        p = self.p
        phi0 = self.phi0
        phi_e = self.phi_end
        if p == 2:
            return (phi0**2 * (-np.log(phi0 - phi) + np.log(phi0 - phi_e)) / 4
                    + (phi0 - phi)**2 / 8 - (phi0 - phi_e)**2 / 8)
        else:
            return (+ phi0**p * (phi0 - phi)**(2 - p) / (2 * (p - 2))
                    - phi0**p * (phi0 - phi_e)**(2 - p) / (2 * (p - 2))
                    + (phi0 - phi)**2 / 4
                    - (phi0 - phi_e)**2 / 4) / p

    def sr_N2phi(self, N):  # noqa: D102
        phi0 = self.phi0
        phis = np.linspace(0, phi0, 100001)[1:-1]
        Ns = self.sr_phi2N(phis)
        N2phi = interp1d(Ns, phis)
        return N2phi(N)


class DoubleWell2Potential(DoubleWellPotential):
    """Quadratic Double-Well potential: `V(phi) = Lambda**4 * (1 - (phi/phi0)**2)**2`.

    Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
    """

    tag = 'dw2'
    name = 'DoubleWell2Potential'
    tex = r'Double-Well (quadratic)'
    perturbation_ic = (1, 0, 0, 1)

    def __init__(self, **pot_kwargs):
        super().__init__(p=2, **pot_kwargs)

    @cached_property
    def phi_end(self):  # noqa: D102
        phi0 = self.phi0
        phi_0_phi_end = np.sqrt(phi0**2 - 2 * np.sqrt(2) * np.sqrt(phi0**2 + 2) + 4) / phi0
        return phi0 * (1 - phi_0_phi_end)

    @staticmethod
    def phi2efolds(phi_shifted, phi0):
        """Get e-folds `N` from inflaton `phi`.

        Find the number of e-folds `N` till end of inflation from inflaton `phi`
        using the slow-roll approximation.

        Parameters
        ----------
        phi_shifted : float or np.ndarray
            Inflaton field `phi` shifted by phi0 such that left potential
            minimum is at zero.
        phi0 : float
            Inflaton distance between local maximum and minima.

        Returns
        -------
        N : float or np.ndarray
            Number of e-folds `N` until end of inflation.

        """
        assert np.all(phi_shifted < phi0)
        phi2 = (phi_shifted - phi0)**2
        phi_end2 = 4 + phi0**2 - 2 * np.sqrt(4 + 2 * phi0**2)
        return (phi2 - phi_end2 - phi0**2 * np.log(phi2 / phi_end2)) / 8


class DoubleWell4Potential(DoubleWellPotential):
    """Quartic Double-Well potential: `V(phi) = Lambda**4 * (1 - (phi/phi0)**4)**2`.

    Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
    """

    tag = 'dw4'
    name = 'DoubleWell4Potential'
    tex = r'Double-Well (quartic)'
    perturbation_ic = (1, 0, 0, 1)

    def __init__(self, **pot_kwargs):
        super().__init__(p=4, **pot_kwargs)

    @staticmethod
    def phi_end_squared(phi0):
        """Get inflaton at end of inflation using slow-roll.

        Parameters
        ----------
        phi0 : float
            Inflaton distance between local maximum and minima.

        Returns
        -------
        phi_end2 : float
            Inflaton phi squared at end of inflation. (unshifted!)
        """
        a = (216 * phi0**8 + phi0**12 - 12 * np.sqrt(3. * phi0**16 * (108 + phi0**4)))**(1/3)
        b = 192 + phi0**4 + phi0**8 / a + a
        return (8 + np.sqrt(b) / np.sqrt(3)
                - np.sqrt(128 - (a - phi0**4)**2 / (3 * a)
                          + (8 * np.sqrt(3) * (128 + phi0**4)) / np.sqrt(b)))

    @classmethod
    def phi2efolds(cls, phi_shifted, phi0):
        """Get e-folds `N` from inflaton `phi`.

        Find the number of e-folds `N` till end of inflation from inflaton `phi`
        using the slow-roll approximation.

        Parameters
        ----------
        phi_shifted : float or np.ndarray
            Inflaton field `phi` shifted by phi0 such that left potential
            minimum is at zero.
        phi0 : float
            Inflaton distance between local maximum and minima.

        Returns
        -------
        N : float or np.ndarray
            Number of e-folds `N` until end of inflation.

        """
        assert np.all(phi_shifted < phi0)
        phi2 = (phi_shifted - phi0)**2
        phi_end2 = cls.phi_end_squared(phi0=phi0)
        return (phi2 - phi_end2 + phi0**4 * (1/phi2 - 1/phi_end2)) / 16


class TmodelPotential(InflationaryPotential):
    """T-model potential: `V(phi) = Lambda**4 * (tanh(phi / (sqrt(6) * alpha)))**(2*p)`."""

    tag = 'tmn'
    name = 'TmodelPotential'
    tex = r'T-model'
    perturbation_ic = (1, 0, 0, 1)

    def __init__(self, **pot_kwargs):
        self.n = pot_kwargs.pop('n')
        self.alpha = pot_kwargs.pop('alpha')
        self.s_6_a = np.sqrt(6 * self.alpha)
        super().__init__(**pot_kwargs)

    def V(self, phi):  # noqa: D102
        C = np.tanh(phi / self.s_6_a)
        return self.Lambda**4 * C**(2 * self.n)

    def dV(self, phi):  # noqa: D102
        n = self.n
        L1 = n * self.Lambda**4 / self.s_6_a
        C = np.tanh(phi / self.s_6_a)
        return 2 * L1 * (C**(2*n-1) - C**(2*n+1))

    def d2V(self, phi):  # noqa: D102
        n = self.n
        L2 = 2 * n * self.Lambda**4 / self.s_6_a**2
        C = np.tanh(phi / self.s_6_a)
        return L2 * (+ C**(2*n-2) * (2 * n - 1)
                     - C**(2*n+0) * 4 * n
                     + C**(2*n+2) * (2 * n + 1))

    def d3V(self, phi):  # noqa: D102
        n = self.n
        L3 = 4 * n * self.Lambda**4 / self.s_6_a**3
        C = np.tanh(phi / self.s_6_a)
        return L3 * (+ C**(2*n-3) * (2*n**2 - 3*n + 1)
                     + C**(2*n-1) * (-6*n**2 + 3*n - 1)
                     + C**(2*n+1) * (6*n**2 + 3*n + 1)
                     + C**(2*n+3) * (-2*n**2 - 3*n - 1))

    def d4V(self, phi):  # noqa: D102
        n = self.n
        L4 = 4 * n * self.Lambda**4 / self.s_6_a**4
        C = np.tanh(phi / self.s_6_a)
        return L4 * (+ C**(2*n - 4) * (4*n**3 - 12*n**2 + 11*n - 3)
                     + C**(2*n - 2) * (-16*n**3 + 24*n**2 - 16*n + 4)
                     + C**(2*n + 0) * (24 * n**3 + 10 * n)
                     + C**(2*n + 2) * (-16*n**3 - 24*n**2 - 16*n - 4)
                     + C**(2*n + 4) * (4*n**3 + 12*n**2 + 11*n + 3))

    def inv_V(self, V):  # noqa: D102
        n = self.n
        return self.s_6_a * np.arctanh(V**(1/(2*n)) / self.Lambda**(2/n))

    def get_epsilon_1V(self, phi):  # noqa: D102
        return 8 * self.n**2 / (self.s_6_a**2 * np.sinh(2 * phi / self.s_6_a)**2)

    def get_epsilon_2V(self, phi):  # noqa: D102
        C = np.tanh(phi / self.s_6_a)
        return 4 * self.n * (1 - C**4) / (C**2 * self.s_6_a**2)

    def get_epsilon_3V(self, phi):  # noqa: D102
        C = np.tanh(phi / self.s_6_a)
        return 4 * self.n * (-C**6 + C**4 - C**2 + 1) / (C**2 * self.s_6_a**2 * (C**2 + 1))

    def get_epsilon_4V(self, phi):  # noqa: D102
        C = np.tanh(phi / self.s_6_a)
        return (4 * self.n * (-C**10 - C**8 + 4*C**6 - 4*C**4 + C**2 + 1)
                / (C**2 * self.s_6_a**2 * (C**6 + C**4 + C**2 + 1)))

    @cached_property
    def phi_end(self):  # noqa: D102
        return self.s_6_a/2 * np.arcsinh(2 * np.sqrt(2) * self.n / self.s_6_a)

    def sr_phi2N(self, phi):  # noqa: D102
        n = self.n
        s_6_a = self.s_6_a
        return s_6_a/(8*n) * (s_6_a * np.cosh(2*phi/s_6_a) - np.sqrt(8*n**2 + s_6_a**2))

    def sr_N2phi(self, N):  # noqa: D102
        n = self.n
        s_6_a = self.s_6_a
        return s_6_a / 2 * np.arccosh(8*n/s_6_a**2 * N + np.sqrt(8*n**2+s_6_a**2)/s_6_a)


# TODO:
# class HilltopPotential(InflationaryPotential):
#     """Double-Well potential: `V(phi) = Lambda**4 * (1 - (phi/phi0)**p)**2`.
#
#     Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
#     """
#
#     tag = 'htp'
#     name = 'HilltopPotential'
#     tex = r'Hilltop (p)'
#
#     def __init__(self, **pot_kwargs):
#         self.phi0 = pot_kwargs.pop('phi0')
#         self.p = pot_kwargs.pop('p')
#         super(HilltopPotential, self).__init__(**pot_kwargs)
#         self.prefactor = 2 * self.p * self.Lambda**4
#
#     def V(self, phi):
#         """`V(phi) = Lambda**4 * (1 - (phi/phi0)**p)**2`.
#
#         Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
#         """
#         phi -= self.phi0
#         return self.Lambda**4 * (1 - (phi / self.phi0)**self.p)**2
#
#     def dV(self, phi):
#         """`V'(phi) = 2p*Lambda**4 * (-1 + (phi / phi0)**p) * phi**(p - 1) / phi0**p`.
#
#         Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
#         """
#         p = self.p
#         phi0 = self.phi0
#         pre = self.prefactor
#         phi -= phi0
#         return pre * (-1 + (phi / phi0)**p) * phi**(p - 1) / phi0**p
#
#     def d2V(self, phi):
#         """`V''(phi) = 2p*Lambda**4 * (1-p+(2*p-1)*(phi/phi0)**p) * phi**(p-2) / phi0**p`.
#
#         Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
#         """
#         p = self.p
#         phi0 = self.phi0
#         pre = self.prefactor
#         phi -= phi0
#         return pre * (1 - p + (2 * p - 1) * (phi / phi0)**p) * phi**(p - 2) / phi0**p
#
#     def d3V(self, phi):
#         """`V'''(phi) = 2p(p-1)Lambda**4 * (2-p+(4*p-2)*(phi/phi0)**p) * phi**(p-3) / phi0**p`.
#
#         Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
#         """
#         p = self.p
#         phi0 = self.phi0
#         pre = self.prefactor
#         phi -= phi0
#         return pre * (p - 1) * (2 - p + (4 * p - 2) * (phi / phi0)**p) * phi**(p - 3) / phi0**p
#
#     def inv_V(self, V):
#         """`phi(V) = phi0 * (1 - sqrt(V) / Lambda**2)**(1/p)`."""
#         return self.phi0 * (1 - np.sqrt(V) / self.Lambda**2)**(1/self.p)


class FeatureFunction(ABC):
    """Feature in the inflationary potential."""

    @staticmethod
    @abstractmethod
    def F(x, x0, a, b):
        """Feature function."""

    @staticmethod
    @abstractmethod
    def dF(x, x0, a, b):
        """Feature function derivative."""

    @staticmethod
    @abstractmethod
    def d2F(x, x0, a, b):
        """Feature function 2nd derivative."""

    @staticmethod
    @abstractmethod
    def d3F(x, x0, a, b):
        """Feature function 3rd derivative."""


class GaussianDip(FeatureFunction):
    """Gaussian: `F(x) = -a * exp(-(x-x0)**2 / (2*b**2))`."""

    @staticmethod
    def F(x, x0, a, b):
        """`F(x) = -a * exp(-(x-x0)**2 / (2*b**2))`."""
        return -a * np.exp(-(x - x0)**2 / (2 * b**2))

    @staticmethod
    def dF(x, x0, a, b):
        """`F'(x) = a/b**2 * (x-x0) * exp(-(x-x0)**2 / (2*b**2))`."""
        return a / b**2 * (x - x0) * np.exp(-(x - x0)**2 / (2 * b**2))

    @staticmethod
    def d2F(x, x0, a, b):
        """`F''(x) = a/b**4 * (b**2 - (x-x0)**2) * exp(-(x-x0)**2 / (2*b**2))`."""
        return a / b**4 * (b**2 - (x - x0)**2) * np.exp(-(x - x0)**2 / (2 * b**2))

    @staticmethod
    def d3F(x, x0, a, b):
        """`F'''(x) = a/b**6 * (x-x0) * ((x-x0)**2 - 3*b**2) * exp(-(x-x0)**2 / (2*b**2))`."""
        return a / b**6 * (x - x0) * ((x - x0)**2 - 3 * b**2) * np.exp(-(x - x0)**2 / (2 * b**2))


class TanhStep(FeatureFunction):
    """Tanh step function: `F(x) = a * tanh((x - x0) / b)`."""

    @staticmethod
    def F(x, x0, a, b):
        """`F(x) = a * tanh((x-x0)/b)`."""
        return a * np.tanh((x - x0) / b)

    @staticmethod
    def dF(x, x0, a, b):
        """`F'(x) = a/b * (1 - tanh((x-x0)/b)**2)`."""
        tanh = np.tanh((x - x0) / b)
        return a / b * (1 - tanh**2)

    @staticmethod
    def d2F(x, x0, a, b):
        """`F''(x) = -2*a/b**2 * tanh((x-x0)/b) * (1 - tanh((x-x0)/b)**2)`."""
        tanh = np.tanh((x - x0) / b)
        return -2 * a / b**2 * tanh * (1 - tanh**2)

    @staticmethod
    def d3F(x, x0, a, b):
        """`F'''(x) = -2*a/b**3 * (1 - 4*tanh((x-x0)/b)**2 + 3*tanh((x-x0)/b)**4)`."""
        tanh = np.tanh((x - x0) / b)
        return -2 * a / b**3 * (1 - 4 * tanh**2 + 3 * tanh**4)


class FeaturePotential(InflationaryPotential, FeatureFunction):
    """Inflationary potential with a feature: `V(phi) = V0(phi) * (1+F(phi))`."""

    def __init__(self, **pot_kwargs):
        self.phi_feature = pot_kwargs.pop('phi_feature')  # position of feature
        self.a_feature = pot_kwargs.pop('a_feature')      # e.g. height or amplitude of feature
        self.b_feature = pot_kwargs.pop('b_feature')      # e.g. width or gradient of feature
        super().__init__(**pot_kwargs)

    def V(self, phi):
        """Inflationary potential V0(phi) with a feature function F(phi).

        `V(phi) = V0(phi) * (1 + F(phi))`
        """
        V0 = super().V(phi)
        F = super().F(phi, self.phi_feature, self.a_feature, self.b_feature)
        return V0 * (1 + F)

    def dV(self, phi):
        """First derivative of the inflationary potential with a feature.

        `V'(phi) = V0'(phi) * (1 + F(phi)) + V0(phi) * F'(phi)`
        """
        V0 = super().V(phi)
        dV0 = super().dV(phi)
        F = super().F(phi, self.phi_feature, self.a_feature, self.b_feature)
        dF = super().dF(phi, self.phi_feature, self.a_feature, self.b_feature)
        return dV0 * (1 + F) + V0 * dF

    def d2V(self, phi):
        """Second derivative of the inflationary potential with a feature.

        `V''(phi) = V0''(phi) * (1 + F(phi)) + 2*V0'(phi)*F'(phi) + V0(phi)*F''(phi)`
        """
        V0 = super().V(phi)
        dV0 = super().dV(phi)
        d2V0 = super().d2V(phi)
        F = super().F(phi, self.phi_feature, self.a_feature, self.b_feature)
        dF = super().dF(phi, self.phi_feature, self.a_feature, self.b_feature)
        d2F = super().d2F(phi, self.phi_feature, self.a_feature, self.b_feature)
        return d2V0 * (1 + F) + 2 * dV0 * dF + V0 * d2F

    def d3V(self, phi):
        r"""Third derivative of the inflationary potential with a feature.

        .. math::
            V'''(\phi) = V_0'''(\phi) * (1 + F(\phi))
                        + 3 * V_0''(\phi) * F'(\phi)
                        + 3 * V_0'(\phi) * F''(\phi)
                        + V_0(\phi) * F'''(\phi)
        """
        V0 = super().V(phi)
        dV0 = super().dV(phi)
        d2V0 = super().d2V(phi)
        d3V0 = super().d3V(phi)
        F = super().F(phi, self.phi_feature, self.a_feature, self.b_feature)
        dF = super().dF(phi, self.phi_feature, self.a_feature, self.b_feature)
        d2F = super().d2F(phi, self.phi_feature, self.a_feature, self.b_feature)
        d3F = super().d3F(phi, self.phi_feature, self.a_feature, self.b_feature)
        return d3V0 * (1 + F) + 3 * d2V0 * dF + 3 * dV0 * d2F + V0 * d3F


class StarobinskyGaussianDipPotential(FeaturePotential, StarobinskyPotential, GaussianDip):
    """Starobinsky potential with a Gaussian dip."""

    tag = 'sgd'
    name = 'StarobinskyGaussianDipPotential'
    tex = r'Starobinsky with a Gaussian dip'


class StarobinskyTanhStepPotential(FeaturePotential, StarobinskyPotential, TanhStep):
    """Starobinsky potential with a hyperbolic tangent step."""

    tag = 'sts'
    name = 'StarobinskyTanhStepPotential'
    tex = r'Starobinsky with a tanh step'
