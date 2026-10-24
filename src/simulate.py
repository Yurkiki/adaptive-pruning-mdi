"""Simulated data where the true variable importance is known analytically."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Component:
    """A univariate additive contribution f_j, centred so E[f_j(X_j)] = 0."""

    name: str
    func: Callable[[FloatArray], FloatArray]
    variance: float


def _linear(x: FloatArray) -> FloatArray:
    return np.asarray(x - 0.5, dtype=np.float64)


def _sine(x: FloatArray) -> FloatArray:
    return np.asarray(np.sin(2.0 * np.pi * x), dtype=np.float64)


def _quadratic(x: FloatArray) -> FloatArray:
    return x**2 - 1.0 / 3.0


def _step(x: FloatArray) -> FloatArray:
    return np.where(x > 0.5, 0.5, -0.5)


def _bump(x: FloatArray) -> FloatArray:
    return np.where((x > 0.4) & (x < 0.6), 0.8, -0.2)


def _zero(x: FloatArray) -> FloatArray:
    return np.zeros_like(x)


LINEAR = Component("linear", _linear, 1.0 / 12.0)
SINE = Component("sine", _sine, 0.5)
QUADRATIC = Component("quadratic", _quadratic, 4.0 / 45.0)
STEP = Component("step", _step, 0.25)
BUMP = Component("bump", _bump, 0.16)
NOISE = Component("noise", _zero, 0.0)

SIGNAL_LIBRARY: tuple[Component, ...] = (LINEAR, SINE, QUADRATIC, STEP, BUMP)


@dataclass(frozen=True)
class Scenario:
    name: str
    components: tuple[Component, ...]
    sigma: float

    @property
    def n_features(self) -> int:
        return len(self.components)

    @property
    def true_importance(self) -> FloatArray:
        return np.array([c.variance for c in self.components], dtype=np.float64)

    @property
    def is_noise(self) -> BoolArray:
        return np.asarray(self.true_importance == 0.0, dtype=np.bool_)


def scenario_s1(sigma: float = 0.5, n_noise: int = 5) -> Scenario:
    """Baseline: five smooth signals of comparable size plus pure-noise features."""
    return Scenario("S1", SIGNAL_LIBRARY + (NOISE,) * n_noise, sigma)


def sample(
    scenario: Scenario, n: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    x = rng.random((n, scenario.n_features))
    y = np.zeros(n, dtype=np.float64)
    for j, component in enumerate(scenario.components):
        y += component.func(x[:, j])
    y += rng.normal(0.0, scenario.sigma, size=n)
    return x, y


def monte_carlo_variance(
    component: Component, n: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Estimate Var(f(X)) for X ~ U[0,1], with the standard error of the estimate."""
    values = component.func(rng.random(n))
    centred = values - values.mean()
    estimate = float((centred**2).mean())
    stderr = float(np.sqrt(((centred**2 - estimate) ** 2).mean() / n))
    return estimate, stderr
