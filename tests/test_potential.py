import numpy as np
from primpy.potential import QuadraticPotential, StarobinskyPotential


def test_quadratic_inflation():
    pot = QuadraticPotential(m=1)
    assert pot.V(phi=1) == 0.5
    pot = QuadraticPotential(m=6e-6)
    assert pot.V(phi=20) == 0.5 * 6e-6**2 * 20**2


def test_starobinsky_inflation():
    pot = StarobinskyPotential(Lambda=1)
    assert pot.V(phi=1) == 0.5
    pot = StarobinskyPotential(Lambda=1e-2)
    assert pot.V(phi=15) == 1e-2**4 * (1 - np.exp(np.sqrt(2 / 3) * 15))**2
