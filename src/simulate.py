"""Simulated data where the true variable importance is known analytically."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt
from scipy import integrate

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


def rescaled(component: Component, variance: float) -> Component:
    """Same shape, scaled so that Var(f(X)) equals `variance`."""
    if component.variance <= 0.0:
        raise ValueError("cannot rescale a zero-variance component")
    k = float(np.sqrt(variance / component.variance))
    base = component.func

    def scaled(x: FloatArray) -> FloatArray:
        return k * base(x)

    return Component(f"{component.name}*{k:.4g}", scaled, variance)


@dataclass(frozen=True)
class Scenario:
    name: str
    components: tuple[Component, ...]
    snr: float

    @property
    def n_features(self) -> int:
        return len(self.components)

    @property
    def true_importance(self) -> FloatArray:
        return np.array([c.variance for c in self.components], dtype=np.float64)

    @property
    def is_noise(self) -> BoolArray:
        return np.asarray(self.true_importance == 0.0, dtype=np.bool_)

    @property
    def sigma(self) -> float:
        return float(np.sqrt(self.true_importance.sum() / self.snr))


def scenario_s1(snr: float = 4.0, n_noise: int = 5) -> Scenario:
    """Baseline: five smooth signals of comparable size plus pure-noise features."""
    return Scenario(f"S1_snr{snr:g}", SIGNAL_LIBRARY + (NOISE,) * n_noise, snr)


def scenario_s4(snr: float = 4.0, n_noise: int = 5) -> Scenario:
    """Heterogeneous signal: true importances spanning two orders of magnitude.

    A constant ratio of 3 between consecutive components. The weakest is 0.02
    against a noise variance near 0.6, so it sits close to the detection limit
    deliberately: a good measure should call it small but non-zero rather than
    zero. That also caps the achievable signal-versus-noise AUC in this
    scenario, which is why Kendall tau matters more here.
    """
    targets = (1.62, 0.54, 0.18, 0.06, 0.02)
    signals = tuple(
        rescaled(c, t) for c, t in zip(SIGNAL_LIBRARY, targets, strict=True)
    )
    return Scenario(f"S4_snr{snr:g}", signals + (NOISE,) * n_noise, snr)


# --- Duroux & Scornet (2016), section 4 ------------------------------------
#
# The eight regression models of arXiv:1603.04261, page 9. Covariates are
# uniform on [0,1]^d and the models are written in Xt = 2(X - 0.5), uniform on
# [-1,1]. Only the ADDITIVE ones are reproduced here: models 2, 5 and 7 contain
# products and multi-variable indicators, so Var(f_j(X_j)) is the population MDI
# of no tree and there is nothing to measure bias against.
#
# Noise is set through the SNR here, as in every other scenario, which departs
# from the paper twice and deliberately:
#
#   - models 1 and 8 carry no noise term at all. A noiseless response leaves no
#     sigma^2 for a deep tree to absorb, which is the very bias under study;
#   - model 6's noise is -1{N(0,1) > 1.25}, a Bernoulli of variance ~0.094, not
#     a Gaussian. Only its variance matters for the absorption mechanism, but the
#     distribution is not the paper's.
#
# What the paper's own noise corresponds to in SNR terms is recorded in PROTOCOL.


def _tilde(x: FloatArray) -> FloatArray:
    """The paper's Xt = 2(X - 0.5): uniform on [-1, 1]."""
    return np.asarray(2.0 * x - 1.0, dtype=np.float64)


def from_function(
    name: str,
    raw: Callable[[FloatArray], FloatArray],
    breaks: tuple[float, ...] = (),
) -> Component:
    """Centre a contribution and integrate its variance over X ~ U[0,1].

    `breaks` lists discontinuities so the quadrature does not straddle them.
    """
    points = list(breaks) if breaks else None

    def scalar(power: int) -> float:
        value, _ = integrate.quad(
            lambda t: float(raw(np.array([t]))[0]) ** power,
            0.0, 1.0, points=points, limit=200,
        )
        return float(value)

    mean = scalar(1)
    variance = scalar(2) - mean**2

    def centred(x: FloatArray) -> FloatArray:
        return np.asarray(raw(x) - mean, dtype=np.float64)

    return Component(name, centred, variance)


def _place(d: int, components: dict[int, Component]) -> tuple[Component, ...]:
    """Put each contribution at its own feature index; the rest is noise."""
    if components and max(components) >= d:
        raise ValueError(f"a component sits past the last of {d} features")
    return tuple(components.get(j, NOISE) for j in range(d))


def _ds1_components() -> dict[int, Component]:
    return {
        0: from_function("Xt^2", lambda x: _tilde(x) ** 2),
        1: from_function("exp(-Xt^2)", lambda x: np.exp(-_tilde(x) ** 2)),
    }


def _ds3_components() -> dict[int, Component]:
    return {
        0: from_function("-sin(2Xt)", lambda x: -np.sin(2.0 * _tilde(x))),
        1: from_function("Xt^2", lambda x: _tilde(x) ** 2),
        2: from_function("Xt", _tilde),
        3: from_function("-exp(-Xt)", lambda x: -np.exp(-_tilde(x))),
    }


def _ds4_components() -> dict[int, Component]:
    def wave(x: FloatArray) -> FloatArray:
        t = 2.0 * np.pi * _tilde(x)
        return np.asarray(
            np.sin(t) + 2.0 * np.cos(t) + 3.0 * np.sin(t) ** 2 + 4.0 * np.cos(t) ** 2,
            dtype=np.float64,
        )

    def ratio(x: FloatArray) -> FloatArray:
        t = np.sin(2.0 * np.pi * _tilde(x))
        return np.asarray(t / (2.0 - t), dtype=np.float64)

    return {
        0: from_function("Xt", _tilde),
        1: from_function("(2Xt-1)^2", lambda x: (2.0 * _tilde(x) - 1.0) ** 2),
        2: from_function("sin/(2-sin)", ratio),
        3: from_function("trig", wave),
    }


def _ds6_components() -> dict[int, Component]:
    # 1{Xt^3 < 0} is 1{Xt < 0}: a fair coin, variance exactly 1/4
    step = from_function("1{Xt<0}", lambda x: (_tilde(x) ** 3 < 0).astype(np.float64),
                         breaks=(0.5,))
    return {k: step for k in range(10)}


def _ds8_components() -> dict[int, Component]:
    return {
        0: from_function("Xt", _tilde),
        2: from_function("3Xt^2", lambda x: 3.0 * _tilde(x) ** 2),
        4: from_function("-2exp(-Xt)", lambda x: -2.0 * np.exp(-_tilde(x))),
        5: from_function("Xt", _tilde),
    }


_DUROUX = {
    "DS1": (50, _ds1_components),
    "DS3": (100, _ds3_components),
    "DS4": (100, _ds4_components),
    "DS6": (30, _ds6_components),
    "DS8": (1000, _ds8_components),
}

# the sample size each model is given in the paper
DUROUX_SIZES = {"DS1": 800, "DS3": 600, "DS4": 600, "DS6": 500, "DS8": 500}


def duroux(tag: str, snr: float = 4.0, d: int | None = None) -> Scenario:
    """An additive model of Duroux & Scornet (2016), section 4."""
    if tag not in _DUROUX:
        raise ValueError(f"{tag} is not an additive model; pick from {sorted(_DUROUX)}")
    default_d, build = _DUROUX[tag]
    width = default_d if d is None else d
    return Scenario(f"{tag}_snr{snr:g}", _place(width, build()), snr)


class ScenarioFactory(Protocol):
    def __call__(self, snr: float = ..., n_noise: int = ...) -> Scenario: ...


def _duroux_factory(tag: str) -> ScenarioFactory:
    def make(snr: float = 4.0, n_noise: int = 0) -> Scenario:
        if n_noise:
            raise ValueError(
                f"{tag} takes its feature count from the paper, so n_noise does "
                f"not apply; pass d= to duroux() to narrow it instead"
            )
        return duroux(tag, snr=snr)

    return make


SCENARIOS: dict[str, ScenarioFactory] = {
    "S1": scenario_s1,
    "S4": scenario_s4,
    **{tag: _duroux_factory(tag) for tag in _DUROUX},
}


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
