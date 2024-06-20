#!/usr/bin/env python
""":mod:`primpy.potentials`: inflationary potentials."""
from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import interp1d
from primpy.units import pi


class InflationaryPotential(ABC):
    """Base class for inflaton potential and derivatives."""

    @abstractmethod
    def __init__(self, **pot_kwargs):
        self.Lambda = pot_kwargs.pop('Lambda')
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
    # def sr_phi_end(self):
    #     """Slow-roll approximation for the inflaton value at the end of inflation."""

    # TODO:
    # @abstractmethod
    # def sr_n_s(self):
    #     """Slow-roll approximation for the spectral index."""

    # TODO:
    # @abstractmethod
    # def sr_n_s(self):
    #     """Slow-roll approximation for the tensor-to-scalar ratio."""

    # TODO:
    # @abstractmethod
    # def sr_epsilon(self):
    #     """Slow-roll potential parameter `epsilon`."""

    # TODO:
    # @abstractmethod
    # def sr_eta(self):
    #     """Slow-roll potential parameter `eta`."""

    @classmethod
    @abstractmethod
    def sr_As2Lambda(cls, A_s, phi_star, N_star, **pot_kwargs):
        """Get potential amplitude `Lambda` from PPS amplitude `A_s`."""


class MonomialPotential(InflationaryPotential):
    """Monomial potential: `V(phi) = Lambda**4 * phi**p`."""

    tag = 'mnp'
    name = 'MonomialPotential'
    tex = r'$\phi^p$'
    perturbation_ic = (1e-1, 0, 0, 1e-6)

    def __init__(self, **pot_kwargs):
        self.p = pot_kwargs.pop('p')
        super().__init__(**pot_kwargs)

    def V(self, phi):
        """`V(phi) = Lambda**4 * phi**p`."""
        return self.Lambda**4 * np.abs(phi)**self.p

    def dV(self, phi):
        """`V(phi) = Lambda**4 * phi**(p-1) * p`."""
        return self.Lambda**4 * np.abs(phi)**(self.p - 1.) * self.p

    def d2V(self, phi):
        """`V(phi) = Lambda**4 * phi**(p-2) * p * (p-1)`."""
        return self.Lambda**4 * np.abs(phi)**(self.p - 2.) * self.p * (self.p - 1)

    def d3V(self, phi):
        """`V(phi) = Lambda**4 * phi**(p-3) * p * (p-1) * (p-2)`."""
        return self.Lambda**4 * np.abs(phi)**(self.p - 3.) * self.p * (self.p - 1) * (self.p - 2)

    def inv_V(self, V):
        """`phi(V) = (V / Lambda**4)**(1/p)`."""
        return (V / self.Lambda**4)**(1/self.p)

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

    @staticmethod
    def sr_As2Lambda(A_s, phi_star, N_star, **pot_kwargs):
        """Get potential amplitude `Lambda` from PPS amplitude `A_s`.

        Find the inflaton amplitude `Lambda` (4th root of potential amplitude)
        that produces the desired amplitude `A_s` of the primordial power
        spectrum using the slow-roll approximation.

        Parameters
        ----------
            A_s : float or np.ndarray
                Amplitude `A_s` of the primordial power spectrum.
            phi_star : float or None
                Inflaton value at horizon crossing of the pivot scale.
            N_star : float or None
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        Returns
        -------
            Lambda : float or np.ndarray
                Amplitude parameter `Lambda` for the Monomial potential.
            phi_star : float
                Inflaton value at horizon crossing of the pivot scale.
            N_star: float
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        """
        p = pot_kwargs.pop('p')
        if N_star is None:
            N_star = phi_star**2 / (2 * p) - p / 4
        elif phi_star is None:
            phi_star = np.sqrt(p / 2 * (4 * N_star + p))
        else:
            raise Exception("Need to specify either N_star or phi_star. "
                            "The respective other should be None.")
        Lambda = (3 * A_s)**(1/4) * np.sqrt(2 * pi * p) * phi_star**(-1/2-p/4)
        return Lambda, phi_star, N_star


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

    @staticmethod
    def sr_As2Lambda(A_s, phi_star, N_star, **pot_kwargs):
        """Get potential amplitude `Lambda` from PPS amplitude `A_s`.

        Find the inflaton amplitude `Lambda` (4th root of potential amplitude)
        that produces the desired amplitude `A_s` of the primordial power
        spectrum using the slow-roll approximation.

        Parameters
        ----------
            A_s : float or np.ndarray
                Amplitude `A_s` of the primordial power spectrum.
            phi_star : float or None
                Inflaton value at horizon crossing of the pivot scale.
            N_star : float or None
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        Returns
        -------
            Lambda : float or np.ndarray
                Amplitude parameter `Lambda` for the Linear potential.
            phi_star : float
                Inflaton value at horizon crossing of the pivot scale.
            N_star: float
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        """
        return MonomialPotential.sr_As2Lambda(A_s, phi_star, N_star, p=1)


class QuadraticPotential(MonomialPotential):
    """Quadratic potential: `V(phi) = Lambda**4 * phi**2`."""

    tag = 'mn2'
    name = 'QuadraticPotential'
    tex = r'$\phi^2$'
    perturbation_ic = (1e-1, 0, 0, 1e-5)

    def __init__(self, **pot_kwargs):
        if 'mass' in pot_kwargs:
            raise ValueError("'mass' was dropped use 'Lambda' instead: Lambda**4=mass**2")
        super().__init__(p=2, **pot_kwargs)

    def V(self, phi):
        """`V(phi) = Lambda**4 * phi**2`."""
        return self.Lambda**4 * phi**2

    def dV(self, phi):
        """`V'(phi) = 2 * Lambda**4 * phi`."""
        return 2 * self.Lambda**4 * phi

    def d2V(self, phi):
        """`V''(phi) = 2 * Lambda**4`."""
        return 2 * self.Lambda**4

    def d3V(self, phi):
        """`V'''(phi) = 0`."""
        return 0

    def inv_V(self, V):
        """`phi(V) = sqrt(V) / Lambda**2`."""
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

    @staticmethod
    def sr_As2Lambda(A_s, phi_star, N_star, **pot_kwargs):
        """Get potential amplitude `Lambda` from PPS amplitude `A_s`.

        Find the inflaton mass `m` (i.e. essentially the potential amplitude)
        that produces the desired amplitude `A_s` of the primordial power
        spectrum using the slow roll approximation.

        Parameters
        ----------
            A_s : float or np.ndarray
                Amplitude `A_s` of the primordial power spectrum.
            phi_star : float or None
                Inflaton value at horizon crossing of the pivot scale.
            N_star : float or None
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        Returns
        -------
            Lambda : float or np.ndarray
                Amplitude parameter `Lambda` for the Quadratic potential (Lambda**2 = mass).
            phi_star : float
                Inflaton value at horizon crossing of the pivot scale.
            N_star: float
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        """
        if N_star is None:
            N_star = (phi_star**2 - 2) / 4
        elif phi_star is None:
            phi_star = np.sqrt(4 * N_star + 2)
        else:
            raise Exception("Need to specify either N_star or phi_star. "
                            "The respective other should be None.")
        Lambda = 2 * np.sqrt(pi * np.sqrt(6 * A_s)) / phi_star
        return Lambda, phi_star, N_star


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

    @staticmethod
    def sr_As2Lambda(A_s, phi_star, N_star, **pot_kwargs):
        """Get potential amplitude `Lambda` from PPS amplitude `A_s`.

        Find the inflaton amplitude `Lambda` (4th root of potential amplitude)
        that produces the desired amplitude `A_s` of the primordial power
        spectrum using the slow-roll approximation.

        Parameters
        ----------
            A_s : float or np.ndarray
                Amplitude `A_s` of the primordial power spectrum.
            phi_star : float or None
                Inflaton value at horizon crossing of the pivot scale.
            N_star : float or None
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        Returns
        -------
            Lambda : float or np.ndarray
                Amplitude parameter `Lambda` for the Linear potential.
            phi_star : float
                Inflaton value at horizon crossing of the pivot scale.
            N_star: float
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        """
        return MonomialPotential.sr_As2Lambda(A_s, phi_star, N_star, p=3)


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

    @staticmethod
    def sr_As2Lambda(A_s, phi_star, N_star, **pot_kwargs):
        """Get potential amplitude `Lambda` from PPS amplitude `A_s`.

        Find the inflaton amplitude `Lambda` (4th root of potential amplitude)
        that produces the desired amplitude `A_s` of the primordial power
        spectrum using the slow-roll approximation.

        Parameters
        ----------
            A_s : float or np.ndarray
                Amplitude `A_s` of the primordial power spectrum.
            phi_star : float or None
                Inflaton value at horizon crossing of the pivot scale.
            N_star : float or None
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        Returns
        -------
            Lambda : float or np.ndarray
                Amplitude parameter `Lambda` for the Linear potential.
            phi_star : float
                Inflaton value at horizon crossing of the pivot scale.
            N_star: float
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        """
        return MonomialPotential.sr_As2Lambda(A_s, phi_star, N_star, p=4)


class StarobinskyPotential(InflationaryPotential):
    """Starobinsky potential: `V(phi) = Lambda**4 * (1 - exp(-sqrt(2/3) * phi))**2`."""

    tag = 'stb'
    name = 'StarobinskyPotential'
    tex = r'Starobinsky'
    gamma = np.sqrt(2 / 3)
    perturbation_ic = (1-2, 0, 0, 1e-8)

    def __init__(self, **pot_kwargs):
        super().__init__(**pot_kwargs)

    def V(self, phi):
        """`V(phi) = Lambda**4 * (1 - exp(-sqrt(2/3) * phi))**2`."""
        return self.Lambda**4 * (1 - np.exp(-StarobinskyPotential.gamma * phi))**2

    def dV(self, phi):
        """`V'(phi) = Lambda**4 * 2 * gamma * exp(-2 * gamma * phi) * (-1 + exp(gamma * phi))`."""
        gamma = StarobinskyPotential.gamma
        return self.Lambda**4 * 2 * gamma * np.exp(-2 * gamma * phi) * (np.exp(gamma * phi) - 1)

    def d2V(self, phi):
        """`V''(phi) = Lambda**4 * 2 * gamma**2 * exp(-2*gamma*phi) * (2 - exp(gamma*phi))`."""
        gamma = StarobinskyPotential.gamma
        return self.Lambda**4 * 2 * gamma**2 * np.exp(-2 * gamma * phi) * (2 - np.exp(gamma * phi))

    def d3V(self, phi):
        """`V'''(phi) = Lambda**4 * 2 * gamma**3 * exp(-2*gamma*phi) * (-4 + exp(gamma*phi))`."""
        gamma = StarobinskyPotential.gamma
        return self.Lambda**4 * 2 * gamma**3 * np.exp(-2 * gamma * phi) * (np.exp(gamma * phi) - 4)

    def inv_V(self, V):
        """`phi(V) = -np.log(1 - np.sqrt(V) / Lambda**2) / gamma`."""
        return -np.log(1 - np.sqrt(V) / self.Lambda**2) / StarobinskyPotential.gamma

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

    @classmethod
    def sr_As2Lambda(cls, A_s, phi_star, N_star, **pot_kwargs):
        """Get potential amplitude `Lambda` from PPS amplitude `A_s`.

        Find the inflaton amplitude `Lambda` (4th root of potential amplitude)
        that produces the desired amplitude `A_s` of the primordial power
        spectrum using the slow-roll approximation.

        Parameters
        ----------
            A_s : float or np.ndarray
                Amplitude `A_s` of the primordial power spectrum.
            phi_star : float or None
                Inflaton value at horizon crossing of the pivot scale.
            N_star : float or None
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        Returns
        -------
            Lambda : float or np.ndarray
                Amplitude parameter `Lambda` for the Starobinsky potential.
            phi_star : float
                Inflaton value at horizon crossing of the pivot scale.
            N_star : float
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        """
        if N_star is None:
            N_star = cls.phi2efolds(phi=phi_star)
        elif phi_star is None:
            # phi = np.sqrt(3 / 2) * np.log(4 / 3 * N_star + 1)  # insufficient approximation
            phi_sample = np.linspace(0.95, 20, 100000)  # 0.95 >~ phi_end =~ 0.9402
            N_sample = cls.phi2efolds(phi=phi_sample)
            logN2phi = interp1d(np.log(N_sample), phi_sample)
            phi_star = logN2phi(np.log(N_star))
        else:
            raise Exception("Need to specify either N_star or phi_star. "
                            "The respective other should be None.")
        Lambda = (2 * A_s)**(1/4) * np.sqrt(pi) / np.sinh(phi_star / np.sqrt(6))
        return Lambda, phi_star, N_star


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
    perturbation_ic = (1e-1, 0, 0, 1e-5)

    def __init__(self, **pot_kwargs):
        self.phi0 = pot_kwargs.pop('phi0')
        super().__init__(**pot_kwargs)

    def V(self, phi):
        """`V(phi) = Lambda**4 * (1 - cos(pi*phi/phi0))`."""
        return self.Lambda**4 / 2 * (1 - np.cos(pi * phi / self.phi0))

    def dV(self, phi):
        """`V(phi) = Lambda**4 * sin(pi*phi/phi0) * pi / phi0`."""
        return self.Lambda**4 / 2 * np.sin(pi * phi / self.phi0) * pi / self.phi0

    def d2V(self, phi):
        """`V(phi) = Lambda**4 * cos(pi*phi/phi0) * (pi / phi0)**2`."""
        return self.Lambda**4 / 2 * np.cos(pi * phi / self.phi0) * (pi / self.phi0)**2

    def d3V(self, phi):
        """`V(phi) = -Lambda**4 * sin(pi*phi/phi0) * (pi / phi0)**3`."""
        return -self.Lambda**4 / 2 * np.sin(pi * phi / self.phi0) * (pi / self.phi0)**3

    def inv_V(self, V):
        """`phi(V) = arccos(1 - V / Lambda**4) * phi0 / pi`."""
        return np.arccos(1 - 2 * V / self.Lambda**4) * self.phi0 / pi

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

    @classmethod
    def sr_As2Lambda(cls, A_s, phi_star, N_star, **pot_kwargs):
        """Get potential amplitude `Lambda` from PPS amplitude `A_s`.

        Find the inflaton amplitude `Lambda` (4th root of potential amplitude)
        that produces the desired amplitude `A_s` of the primordial power
        spectrum using the slow-roll approximation.

        Parameters
        ----------
            A_s : float or np.ndarray
                Amplitude `A_s` of the primordial power spectrum.
            phi_star : float or None
                Inflaton value at horizon crossing of the pivot scale.
            N_star : float or None
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        Keyword Arguments
        -----------------
            phi0 : float
                Inflaton distance between local maximum and minima.

        Returns
        -------
            Lambda : float or np.ndarray
                Amplitude parameter `Lambda` for the Natural inflation potential.
            phi_star : float
                Inflaton value at horizon crossing of the pivot scale.
            N_star : float
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        """
        phi0 = pot_kwargs.pop('phi0')
        f = phi0 / pi
        if N_star is None:
            N_star = cls.phi2efolds(phi=phi_star, phi0=phi0)
        elif phi_star is None:
            phi_star = 2 * f * np.arccos(np.exp(-N_star / (2*f**2)) / np.sqrt(1 + 1 / (2*f**2)))
        else:
            raise Exception("Need to specify either N_star or phi_star. "
                            "The respective other should be None.")
        numerator = pi * np.sin(phi_star / f)
        denominator = f * np.sin(phi_star / (2 * f))**3
        Lambda = (3 * A_s)**(1/4) * np.sqrt(numerator / denominator)
        return Lambda, phi_star, N_star


class DoubleWellPotential(InflationaryPotential):
    """Double-Well potential: `V(phi) = Lambda**4 * (1 - (phi/phi0)**p)**2`.

    Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
    """

    tag = 'dwp'
    name = 'DoubleWellPotential'
    tex = r'Double-Well (p)'
    perturbation_ic = (1e-1, 0, 0, 1e-5)

    def __init__(self, **pot_kwargs):
        self.phi0 = pot_kwargs.pop('phi0')
        self.p = pot_kwargs.pop('p')
        super().__init__(**pot_kwargs)
        self.prefactor = 2 * self.p * self.Lambda**4

    def V(self, phi):
        """`V(phi) = Lambda**4 * (1 - (phi/phi0)**p)**2`.

        Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
        """
        return self.Lambda**4 * (1 - ((phi - self.phi0) / self.phi0)**self.p)**2

    def dV(self, phi):
        """`V'(phi) = 2p*Lambda**4 * (-1 + (phi / phi0)**p) * phi**(p - 1) / phi0**p`.

        Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
        """
        p = self.p
        phi0 = self.phi0
        pre = self.prefactor
        return pre * (-1 + ((phi - phi0) / phi0)**p) * (phi - phi0)**(p - 1) / phi0**p

    def d2V(self, phi):
        """`V''(phi) = 2p*Lambda**4 * (1-p+(2*p-1)*(phi/phi0)**p) * phi**(p-2) / phi0**p`.

        Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
        """
        p = self.p
        phi0 = self.phi0
        pre = self.prefactor
        return pre * (1 - p + (2 * p - 1) * ((phi-phi0) / phi0)**p) * (phi-phi0)**(p-2) / phi0**p

    def d3V(self, phi):
        """`V'''(phi) = 2p(p-1)Lambda**4 * (2-p+(4*p-2)*(phi/phi0)**p) * phi**(p-3) / phi0**p`.

        Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
        """
        p = self.p
        phi0 = self.phi0
        pre = self.prefactor
        return pre * (p-1) * (2 - p + (4*p-2) * ((phi-phi0)/phi0)**p) * (phi-phi0)**(p-3) / phi0**p

    def inv_V(self, V):
        """`phi(V) = phi0 * (1 - sqrt(V) / Lambda**2)**(1/p)`."""
        return self.phi0 * (1 - np.sqrt(V) / self.Lambda**2)**(1/self.p)

    @staticmethod
    def sr_As2Lambda(A_s, phi_star, N_star, **pot_kwargs):
        """Get potential amplitude `Lambda` from PPS amplitude `A_s`."""
        raise NotImplementedError("This function is not implemented for DoubleWellPotential, yet. "
                                  "It is implemented for DoubleWell2Potential and "
                                  "DoubleWell4Potential, though. Feel free to raise an issue on"
                                  "github if this is something you need.")


class DoubleWell2Potential(DoubleWellPotential):
    """Quadratic Double-Well potential: `V(phi) = Lambda**4 * (1 - (phi/phi0)**2)**2`.

    Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
    """

    tag = 'dw2'
    name = 'DoubleWell2Potential'
    tex = r'Double-Well (quadratic)'
    perturbation_ic = (1e-1, 0, 0, 1e-5)

    def __init__(self, **pot_kwargs):
        super().__init__(p=2, **pot_kwargs)

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

    @classmethod
    def sr_As2Lambda(cls, A_s, phi_star, N_star, **pot_kwargs):
        """Get potential amplitude `Lambda` from PPS amplitude `A_s`.

        Find the inflaton amplitude `Lambda` (4th root of potential amplitude)
        that produces the desired amplitude `A_s` of the primordial power
        spectrum using the slow-roll approximation.

        Parameters
        ----------
            A_s : float or np.ndarray
                Amplitude `A_s` of the primordial power spectrum.
            phi_star : float or None
                Inflaton value at horizon crossing of the pivot scale.
            N_star : float or None
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        Keyword Arguments
        -----------------
            phi0 : float
                Inflaton distance between local maximum and minima.

        Returns
        -------
            Lambda : float or np.ndarray
                Amplitude parameter `Lambda` for the Double-Well potential.
            phi_star : float
                Inflaton value at horizon crossing of the pivot scale.
            N_star : float
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        """
        phi0 = pot_kwargs.pop('phi0')
        if N_star is None:
            N_star = cls.phi2efolds(phi_shifted=phi_star, phi0=phi0)
        elif phi_star is None:
            # phi = phi_end * np.exp((-8 * N_star - phi_end**2) / (2 * phi0**2))  # inaccurate
            phi_end_shifted = phi0 - np.sqrt(4 + phi0**2 - 2 * np.sqrt(4 + 2 * phi0**2))
            phi_sample = np.linspace(phi_end_shifted, phi0, 100000)[1:-1]
            N_sample = cls.phi2efolds(phi_shifted=phi_sample, phi0=phi0)
            logN2phi = interp1d(np.log(N_sample), phi_sample)
            phi_star = float(logN2phi(np.log(N_star)))
        else:
            raise Exception("Need to specify either N_star or phi_star. "
                            "The respective other should be None.")
        Lambda = (3 * A_s)**(1/4) * np.sqrt(8 * pi * phi_star) * phi0 / (phi0**2 - phi_star**2)
        return Lambda, phi_star, N_star


class DoubleWell4Potential(DoubleWellPotential):
    """Quartic Double-Well potential: `V(phi) = Lambda**4 * (1 - (phi/phi0)**4)**2`.

    Double-Well shifted such that left minimum is at zero: phi -> phi-phi0
    """

    tag = 'dw4'
    name = 'DoubleWell4Potential'
    tex = r'Double-Well (quartic)'
    perturbation_ic = (1e-1, 0, 0, 1e-5)

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

    @classmethod
    def sr_As2Lambda(cls, A_s, phi_star, N_star, **pot_kwargs):
        """Get potential amplitude `Lambda` from PPS amplitude `A_s`.

        Find the inflaton amplitude `Lambda` (4th root of potential amplitude)
        that produces the desired amplitude `A_s` of the primordial power
        spectrum using the slow-roll approximation.

        Parameters
        ----------
            A_s : float or np.ndarray
                Amplitude `A_s` of the primordial power spectrum.
            phi_star : float or None
                Inflaton value at horizon crossing of the pivot scale.
            N_star : float or None
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        Keyword Arguments
        -----------------
            phi0 : float
                Inflaton distance between local maximum and minima.

        Returns
        -------
            Lambda : float or np.ndarray
                Amplitude parameter `Lambda` for the Double-Well potential.
            phi_star : float
                Inflaton value at horizon crossing of the pivot scale.
            N_star : float
                Number of observable e-folds of inflation `N_star`
                from horizon crossing till the end of inflation.

        """
        phi0 = pot_kwargs.pop('phi0')
        if N_star is None:
            N_star = cls.phi2efolds(phi_shifted=phi_star, phi0=phi0)
        elif phi_star is None:
            phi_end_shifted = phi0 - np.sqrt(cls.phi_end_squared(phi0=phi0))
            phi_sample = np.linspace(phi_end_shifted, phi0, 100000)[1:-1]
            N_sample = cls.phi2efolds(phi_shifted=phi_sample, phi0=phi0)
            logN2phi = interp1d(np.log(N_sample), phi_sample)
            phi_star = float(logN2phi(np.log(N_star)))
        else:
            raise Exception("Need to specify either N_star or phi_star. "
                            "The respective other should be None.")
        Lambda = (768 * A_s)**(1/4) * np.sqrt(pi * phi_star**3) * phi0**2 / (phi0**4 - phi_star**4)
        return Lambda, phi_star, N_star


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
