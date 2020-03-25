#!/usr/bin/env python
""":mod:`primpy.potentials`: inflationary potentials."""
import numpy as np
from scipy.interpolate import interp1d
from primpy.units import pi


class InflationaryPotential(object):
    """Base class for inflaton potential and derivatives."""

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
        pass

    def dV(self, phi):
        """First derivative `V'(phi)` w.r.t. inflaton `phi`.

        Parameters
        ----------
        phi : float or np.ndarray
            Inflaton field `phi`.

        Returns
        -------
        dV : float or np.ndarray
            1st derivative of inflationary potential: `V'(phi)`.

        """
        pass

    def d2V(self, phi):
        """Second derivative `V''(phi)` w.r.t. inflaton `phi`.

        Parameters
        ----------
        phi : float or np.ndarray
            Inflaton field `phi`.

        Returns
        -------
        d2V : float or np.ndarray
            2nd derivative of inflationary potential: `V''(phi)`.

        """
        pass

    def d3V(self, phi):
        """Third derivative `V'''(phi)` w.r.t. inflaton `phi`.

        Parameters
        ----------
        phi : float or np.ndarray
            Inflaton field `phi`.

        Returns
        -------
        d3V : float or np.ndarray
            3rd derivative of inflationary potential: `V'''(phi)`.

        """
        pass

    def inv_V(self, V):
        """Inverse function `phi(V)` w.r.t. potential `V`.

        Parameters
        ----------
        V : float or np.ndarray
            Inflationary potential `V`.

        Returns
        -------
        phi : float or np.ndarray
            Inflaton field `phi`.

        """
        pass


class QuadraticPotential(InflationaryPotential):
    """Quadratic potential: `V(phi) = 0.5 * m**2 * phi**2`."""

    def __init__(self, m):
        self.m = m
        super(QuadraticPotential, self).__init__()

    def V(self, phi):
        """`V(phi) = 0.5 * m**2 * phi**2`."""
        return self.m**2 * phi**2 / 2

    def dV(self, phi):
        """`V'(phi) = m**2 phi`."""
        return self.m**2 * phi

    def d2V(self, phi):
        """`V''(phi) = m**2`."""
        return self.m**2

    def d3V(self, phi):
        """`V'''(phi) = 0`."""
        return 0

    def inv_V(self, V):
        """`phi(V) = sqrt(2 * V) / m`."""
        return np.sqrt(2 * V) / self.m

    @staticmethod
    def As2mass(A_s, N_star):
        """Get mass `m` from amplitude `A_s`.

        Find the inflaton mass `m` that produces the desired amplitude `A_s`
        using the slow roll approximation.

        Parameters
        ----------
        A_s : float or np.ndarray
            Amplitude `A_s` of the primordial power spectrum.
        N_star : float
            Number of observable e-folds of inflation `N_star`
            from horizon crossing till the end of inflation.

        Returns
        -------
        m : float or np.ndarray
            Inflaton mass `m` for the quadratic potential.

        """
        phi_star2 = 4 * N_star + 2
        return 4 * pi * np.sqrt(6 * A_s) / phi_star2, np.sqrt(phi_star2)


class StarobinskyPotential(InflationaryPotential):
    """Starobinsky potential: `V(phi) = Lambda**4 * (1 - exp(-sqrt(2/3) * phi))**2`."""

    gamma = np.sqrt(2 / 3)

    def __init__(self, Lambda):
        self.Lambda = Lambda
        super(StarobinskyPotential, self).__init__()

    def V(self, phi):
        """`V(phi) = Lambda**4 * (1 - exp(-sqrt(2/3) * phi))**2`."""
        return self.Lambda ** 4 * (1 - np.exp(-StarobinskyPotential.gamma * phi)) ** 2

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
                - gamma * (phi - phi_end)) / (2 * gamma ** 2)

    @staticmethod
    def As2Lambda(A_s, N_star):
        """Get amplitude `Lambda` from amplitude `A_s`.

        Find the inflaton amplitude `Lambda` that produces the desired
        amplitude `A_s` using the slow-roll approximation.

        Parameters
        ----------
        A_s : float or np.ndarray
            Amplitude `A_s` of the primordial power spectrum.
        N_star : float
            Number of observable e-folds of inflation `N_star`
            from horizon crossing till the end of inflation.

        Returns
        -------
        Lambda : float or np.ndarray
            Amplitude parameter `Lambda` for the Starobinsky potential.

        """
        phi_interpolation_sample = np.linspace(0.95, 20, 10000)  # 0.95 >~ phi_end =~ 0.9402
        N_interpolation_sample = StarobinskyPotential.phi2efolds(phi_interpolation_sample)
        logN2phi = interp1d(np.log(N_interpolation_sample), phi_interpolation_sample)
        phi_star = logN2phi(np.log(N_star))
        # phi = np.sqrt(3 / 2) * np.log(4 / 3 * N_star + 1)  # insufficient approximation
        Lambda2 = np.sqrt(2 * A_s) * pi / np.sinh(phi_star / np.sqrt(6))**2
        return np.sqrt(Lambda2), phi_star
