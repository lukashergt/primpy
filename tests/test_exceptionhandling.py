#!/usr/bin/env python
"""Tests for `primpy.exceptionhandling` module."""
import pytest
import numpy as np
from warnings import warn
from scipy.integrate import solve_ivp
from primpy.exceptionhandling import PrimpyError, StepSizeError, BigBangError
from primpy.exceptionhandling import PrimpyWarning, BigBangWarning, CollapseWarning
from primpy.exceptionhandling import InflationWarning, InflationStartWarning, InflationEndWarning


@pytest.mark.parametrize('Error', [StepSizeError, BigBangError])
def test_PrimpyError(Error):
    with pytest.raises(PrimpyError):
        raise Error("primpy error")


def test_StepSizeError():
    with pytest.raises(StepSizeError):
        raise StepSizeError("step too small")


def test_BigBangError():
    with pytest.raises(BigBangError):
        raise BigBangError("no Big Bang")


@pytest.mark.parametrize('CustomWarning', [BigBangWarning, InflationWarning, CollapseWarning,
                                           InflationStartWarning, InflationEndWarning])
def test_PrimpyWarning(CustomWarning):
    with pytest.warns(PrimpyWarning):
        warn(CustomWarning("primpy warning"))


@pytest.mark.parametrize('CustomWarning', [InflationStartWarning, InflationEndWarning,
                                           CollapseWarning])
def test_InflationWarning(CustomWarning):
    with pytest.warns(InflationWarning):
        warn(CustomWarning("inflation warning"))


def test_InflationStartWarning():
    with pytest.warns(InflationStartWarning):
        warn(InflationStartWarning("inflation start warning", events={}))
    with pytest.warns(InflationStartWarning):
        warn(InflationStartWarning("inflation start warning", events={'Inflation_dir+1_term0': 0}))


# noinspection PyUnresolvedReferences
def test_InflationEndWarning():
    with pytest.warns(InflationEndWarning):
        warn(InflationEndWarning("warning with empty events", events={}))
    with pytest.warns(InflationEndWarning):
        warn(InflationEndWarning("warning with events", events={'Inflation_dir-1_term1': 0}))
    with pytest.warns(InflationEndWarning):
        sol = solve_ivp(lambda x, y: y, (0, 0), y0=np.zeros(2))
        sol.w = sol.y[0]
        warn(InflationEndWarning("warning with sol", events={'Inflation_dir-1_term1': 0}, sol=sol))


def test_BigBangWarning():
    with pytest.warns(BigBangWarning):
        warn(BigBangWarning("universe might recollapse"))
