#!/usr/bin/env python
""":mod:`primpy.exceptions`: Custom exceptions and warnings."""


class PrimpyError(Exception):
    """Base class for exceptions in primpy."""


class InflationStartError(PrimpyError):
    """Exception when the inflation start condition for open or closed universes is violated.

    Attributes
    ----------
        message : str
            Explanation of the error.

    kwargs
    ------
        geometry : str
            Should be either 'open' or 'closed' to relate to the respective
            condition at inflation start.
            default : "all types of"
    """

    def __init__(self, message, *args, **kwargs):
        self.geometry = kwargs.pop('geometry', "all types of")
        self.header = "Inflation start condition for %s universes is violated." % self.geometry
        self.message = "%s %s" % (self.header, message)
        super(InflationStartError, self).__init__(self.message, *args)


class StepSizeError(PrimpyError):
    """Warning when the scipy integrator failed because of a too small step size.

    Attributes
    ----------
        message : str
            Explanation of the warning.
    """

    def __init__(self, message, *args):
        self.message = message
        super(StepSizeError, self).__init__(message, *args)


class BigBangError(PrimpyError):
    """Exceptions for the standard Big Bang evolution.

    Attributes
    ----------
        message : str
            Explanation of the error.
    """

    def __init__(self, message, *args):
        self.message = message
        super(BigBangError, self).__init__(self.message, *args)


class PrimpyWarning(UserWarning):
    """Base class for warnings in primpy."""


class InflationWarning(PrimpyWarning):
    """Warnings for the inflationary background evolution.

    Attributes
    ----------
        message : str
            Explanation of the warning.
    """

    def __init__(self, message, *args):
        self.message = message
        super(InflationWarning, self).__init__(message, *args)


class CollapseWarning(InflationWarning):
    """Warning when the Universe has collapsed.

    Attributes
    ----------
        message : str
            Explanation of the warning.
    """

    def __init__(self, message, *args):
        self.header = "Universe has collapsed."
        self.message = "%s %s" % (self.header, message)
        super(CollapseWarning, self).__init__(self.message, *args)


class InflationStartWarning(InflationWarning):
    """Warnings when the start of inflation could not be determined.

    Attributes
    ----------
        message : str
            Explanation of the warning.

    kwargs
    ------
        events : dict
            Dictionary of events captured. Can be any of `N_events`, `t_events`, `phi_events`.
            default : None
    """

    def __init__(self, message, *args, **kwargs):
        self.header = "Inflation start not determined."
        events = kwargs.pop('events', None)
        if events is None:
            self.message = "%s. %s" % (self.header, message)
        elif 'Inflation_dir+1_term1' not in events and 'Inflation_dir+1_term0' not in events:
            extra_info = ("Not tracking `InflationEvent`. Events tracked are %s. In order to "
                          "determine the start of inflation `_N_beg`, make sure to track the "
                          "event `InflationEvent(equations, direction=+1)` defined in "
                          "`primpy.events`" % events)
            self.message = "%s: %s. %s" % (self.header, extra_info, message)
        else:
            self.message = "%s. %s" % (self.header, message)
        super(InflationStartWarning, self).__init__(self.message, *args)


class InflationEndWarning(InflationWarning):
    """Warnings when the end of inflation could not be determined.

    Attributes
    ----------
        message : str
            Explanation of the warning.

    kwargs
    ------
        events : dict
            Dictionary of events captured. Can be any of `N_events`, `t_events`, `phi_events`.
            default : None
        sol : Bunch object returned by :func:`primpy.solver.solve`
            default : None
    """

    def __init__(self, message, *args, **kwargs):
        self.header = "Inflation end not determined"
        events = kwargs.pop('events', None)
        sol = kwargs.pop('sol', None)
        if events is None:
            self.message = "%s. %s" % (self.header, message)
        elif 'Inflation_dir-1_term1' not in events and 'Inflation_dir-1_term0' not in events:
            extra_info = ("Not tracking `InflationEvent`. Events tracked are %s. In order to "
                          "determine the end of inflation `_N_end`, make sure to track the event "
                          "`InflationEvent(equations, direction=-1)` defined in `primpy.events`"
                          % events)
            self.message = "%s: %s. %s" % (self.header, extra_info, message)
        elif sol is None:
            self.message = "%s. %s" % (self.header, message)
        elif sol.w[-1] < -1 / 3:
            extra_info = ("Still inflating: N[-1]=%g, phi[-1]=%g, w[-1]=%g"
                          % (sol._N[-1], sol.phi[-1], sol.w[-1]))
            self.message = "%s: %s. %s" % (self.header, extra_info, message)
        else:
            self.message = "%s. %s" % (self.header, message)
        super(InflationEndWarning, self).__init__(self.message, *args)


class BigBangWarning(PrimpyWarning):
    """Warnings for the standard Big Bang evolution.

    Attributes
    ----------
        message : str
            Explanation of the warning.
    """

    def __init__(self, message, *args):
        self.message = message
        super(BigBangWarning, self).__init__(self.message, *args)
