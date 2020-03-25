#!/usr/bin/env python
""":mod:`primpy.equations`: general setup for ODEs."""
from abc import ABC
from types import MethodType
import numpy as np
from scipy.interpolate import interp1d


class Equations(ABC):
    """Base class for equations.

    Allows one to compute derivatives and derived variables.
    Most of the other classes take 'equations' as an object.

    Attributes
    ----------
    idx : dict
        dictionary mapping variable names to indices in the solution vector y

    independent_variable : string
        name of independent variable
    """

    def __init__(self):
        self.idx = {}

    def __call__(self, x, y):
        """Vector of derivatives.

        Parameters
        ----------
        x : float
            independent variable

        y : np.ndarray
            dependent variables

        Returns
        -------
        dy : np.ndarray
            Vector of derivatives
        """
        raise NotImplementedError("Equations class must define __call__")

    def sol(self, sol, **kwargs):
        """Post-processing of `solve_ivp` solution."""
        sol.x = sol.t
        x, idx_x_unique = np.unique(sol.t, return_index=True)
        sol.idx_x_unique = idx_x_unique
        del sol.t
        for name, i in self.idx.items():
            setattr(sol, name, sol.y[i, idx_x_unique])
        x_name = self.independent_variable
        setattr(sol, x_name + '_events', dict(zip(sol.event_keys, sol.pop('t_events'))))
        setattr(sol, x_name, x)
        return sol

    def _set_independent_variable(self, name):
        """Set the name of the independent variable.

        Parameters
        ----------
        name : str
            Name of the independent variable.

        """
        def method(self, x, y):
            return x

        method.__doc__ = """ Hi there """

        setattr(self, name, MethodType(method, self))
        self.independent_variable = name

    def add_variable(self, *args):
        """Add dependent variables to the equations.

        * creates an index for the location of variable in y
        * creates a class method of the same name with signature
          name(self, x, y) that should be used to extract the variable value in
          an index-independent manner.

        Parameters
        ----------
        *args : str
            Name of the dependent variables

        """
        for name in args:
            self._add_variable(name)

    def _add_variable(self, name):
        self.idx[name] = len(self.idx)

        def method(self, x, y):
            return np.array(y)[self.idx[name], ...]

        method.__doc__ = """Retrieve %s from the solution vector.

        Arguments
        ---------
        x : float
            independent variable

        y : np.array
            dependent variables

        Returns
        -------
        %s : float
            value of  %s
        """ % (name, name, name)

        setattr(self, name, MethodType(method, self))

    @staticmethod
    def _interp1d(x, y, **kwargs):
        kind = kwargs.pop('kind', 'cubic')
        bounds_error = kwargs.pop('bounds_error', False)
        return interp1d(x, y, kind=kind, bounds_error=bounds_error, **kwargs)