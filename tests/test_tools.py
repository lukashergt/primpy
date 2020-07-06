#!/usr/bin/env python
"""Tools for pytest tests."""
import pytest


def effequal(expected, rel=1e-15, abs=1e-15, **kwargs):
    return pytest.approx(expected, rel=rel, abs=abs, **kwargs)
