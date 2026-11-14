"""The world the market maker trades in, and the baseline parameter set.

Baseline (Avellaneda and Stoikov 2008, section 3.3):

    dS_t = sigma dW_t,   fills at offset delta arrive at rate  lambda(delta) = A exp(-k delta).

Two optional departures from the model are switched on through ``Market``:

* ``impact`` (adverse selection): market orders arrive on each side at rate A and
  each moves the mid by ``impact`` in its own direction (a buy lifts the mid).
  A fill at my ask is a buy order, so E[dS | ask fill] = +impact.
* ``jump_rate``, ``jump_std``: compound-Poisson N(0, jump_std^2) jumps in the mid,
  independent of order flow.

In both cases ``sigma`` stays the *total* volatility of the mid: the diffusion
variance is reduced so that Var(S_T - S_0) = sigma^2 T. The market maker's sigma
is therefore still right; only the structure of the price moves changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Market:
    s0: float = 100.0
    T: float = 1.0
    sigma: float = 2.0
    A: float = 140.0
    k: float = 1.5
    impact: float = 0.0
    jump_rate: float = 0.0
    jump_std: float = 0.0

    def intensity(self, delta: np.ndarray) -> np.ndarray:
        """lambda(delta) = A exp(-k delta); zero for delta = +inf (no quote)."""
        return self.A * np.exp(-self.k * np.asarray(delta, dtype=float))

    def diffusion_variance(self, dt: float) -> float:
        """Per-unit-time diffusion variance that keeps total variance at sigma^2.

        With Bernoulli(A dt) order arrivals per side the impact term contributes
        2 A (1 - A dt) impact^2 per unit time; jumps contribute jump_rate jump_std^2.
        """
        var = (self.sigma**2
               - 2.0 * self.A * (1.0 - self.A * dt) * self.impact**2
               - self.jump_rate * self.jump_std**2)
        if var < 0:
            raise ValueError("impact/jump variance exceeds sigma^2; "
                             f"impact must be below {self.sigma / np.sqrt(2 * self.A):.4f}")
        return var


# Parameter set of the paper's simulation, plus the choices fixed for this study.
PAPER = Market()
GAMMA = 0.1                 # baseline risk aversion
GAMMAS_PAPER = (0.01, 0.1, 0.5)
DT_PAPER = 0.005            # 200 steps
Q_MAX = 30                  # inventory limit used for the exact solution and all utility comparisons
SEED = 2008
