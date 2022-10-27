#!/usr/bin/env python
""":mod:`primpy.events`: setup for event tracking in :func:`scipy.integrate.solve_ivp`."""
import numpy as np


class Event(object):
    """Base class for event tracking.

    Gives a more usable wrapper to callable event to be passed to
    :func:`scipy.integrate.solve_ivp`.

    Parameters
    ----------
        equations: Equations
            The equations for computing derived variables.

        direction: int [-1, 0, +1], optional, default 0
            The direction of the root finding (if any)

        terminal: bool, optional, default False
            Whether to stop at this root

        value: float, optional, default 0
            Offset to root

    """

    def __init__(self, equations, direction=0, terminal=False, value=0):
        self.equations = equations
        self.direction = direction
        self.terminal = terminal
        self.value = value
        self.name = 'Event'

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
            root : float
                event occurs when this is zero from a given direction

        """
        raise NotImplementedError("Event class must define __call__.")


class UntilTEvent(Event):
    """Stop after a given amount of time `t` has passed."""

    def __init__(self, equations, value, direction=0, terminal=True):
        super(UntilTEvent, self).__init__(equations, direction, terminal, value)
        self.name = 'UntilT'

    def __call__(self, x, y):
        """Root of `t - value`."""
        return self.equations.t(x, y) - self.value


class UntilNEvent(Event):
    """Stop after a given number of e-folds `N`."""

    def __init__(self, equations, value, direction=0, terminal=True):
        super(UntilNEvent, self).__init__(equations, direction, terminal, value)
        self.name = 'UntilN'

    def __call__(self, x, y):
        """Root of `N - value`."""
        return self.equations.N(x, y) - self.value


class InflationEvent(Event):
    """Track inflation start/end."""

    def __init__(self, equations, direction=0, terminal=False, value=0, **kwargs):
        super(InflationEvent, self).__init__(equations, direction, terminal, value)
        self.name = 'Inflation_dir%d_term%d' % (self.direction, self.terminal)
        self.t_i = kwargs.pop('t_i', None)

    def __call__(self, x, y):
        """Root of `V - dphidt**2`."""
        if x == self.t_i:
            return self.direction
        return self.equations.inflating(x, y) - self.value


class AfterInflationEndEvent(Event):
    """Go a bit past the end of inflation."""

    def __init__(self, equations, direction=+1, terminal=True, value=0):
        super(AfterInflationEndEvent, self).__init__(equations, direction, terminal, value)
        self.name = 'AfterInflationEnd_dir%d_term%d' % (self.direction, self.terminal)

    def __call__(self, x, y):
        """Root of `w - value`."""
        return self.equations.w(x, y) - self.value


class CollapseEvent(Event):
    """Stop if Universe collapses, i.e. test whether `H**2` turns negative."""

    def __init__(self, equations, direction=0, terminal=True, value=0):
        super(CollapseEvent, self).__init__(equations, direction, terminal, value)
        self.name = 'Collapse'

    def __call__(self, x, y):
        """Root of `H2 - value`."""
        return self.equations.H2(x, y) - self.value


class Phi0Event(Event):
    """Track zero crossings of inflaton `phi`."""

    def __init__(self, equations, direction=0, terminal=True, value=0):
        super(Phi0Event, self).__init__(equations, direction, terminal, value)
        self.name = 'Phi0_dir%d_term%d' % (self.direction, self.terminal)

    def __call__(self, x, y):
        """Root of `phi - value`."""
        return self.equations.phi(x, y) - self.value


class ModeExitEvent(Event):
    """Track when mode exits the horizon aH."""

    def __init__(self, equations, value, direction=0, terminal=True):
        super(ModeExitEvent, self).__init__(equations, direction, terminal, value)
        self.name = 'ModeExit_dir%d_term%d_%e' % (self.direction, self.terminal, self.value)

    def __call__(self, x, y):
        """Root of `logaH - log(value)`."""
        logH = np.log(np.abs(self.equations.H2(x, y))) / 2
        return logH + self.equations.N(x, y) - np.log(self.value)
