"""Vectorised Monte Carlo simulator of a quoting strategy.

One step of length dt, for every path at once:

  1. quotes (delta^b, delta^a) are computed from tau = T - t and the inventory q;
  2. a bid fill occurs with probability min(1, A exp(-k delta^b) dt) and an ask fill
     with probability min(1, A exp(-k delta^a) dt), independently (at most one each);
     a bid fill does q += 1, cash -= s - delta^b; an ask fill does q -= 1, cash += s + delta^a;
  3. the mid then moves: s += sigma_d sqrt(dt) N(0,1) (+ order-flow impact and jumps
     if the Market has them).

The Bernoulli fill is the first-order approximation of a Poisson arrival and
needs A exp(-k delta) dt << 1; the error is the O((lambda dt)^2) chance of two
arrivals in one step.

Wealth is marked to mid: W = cash + q s, so over a step

    dW = delta^b 1{bid fill} + delta^a 1{ask fill} + q_new * ds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .model import Market
from .quotes import Strategy


@dataclass
class SimResult:
    pnl: np.ndarray             # terminal wealth cash_T + q_T s_T (cash_0 = 0, q_0 = 0)
    inventory: np.ndarray       # q_T
    bid_fills: np.ndarray       # number of bid fills per path
    ask_fills: np.ndarray
    mean_spread: float          # delta^a + delta^b averaged over time and paths
    path: dict | None = None    # full history when record=True


def apply_fills(cash, q, s, delta_b, delta_a, fill_b, fill_a):
    """Cash and inventory after fills at bid s - delta_b and ask s + delta_a."""
    cash = cash - np.where(fill_b, s - delta_b, 0.0) + np.where(fill_a, s + delta_a, 0.0)
    q = q + fill_b.astype(np.int64) - fill_a.astype(np.int64)
    return cash, q


def simulate(market: Market, strategy: Strategy, n_paths: int, n_steps: int, seed: int,
             q_max: int | None = None, record: bool = False) -> SimResult:
    """Simulate ``strategy`` on ``n_paths`` paths of ``n_steps`` steps.

    ``q_max`` imposes a hard inventory limit on any strategy (no bid fill at
    q = q_max, no ask fill at q = -q_max). Diffusion and fill draws come from one
    random stream and jumps from another, so switching impact or jumps off
    reproduces the baseline paths exactly.
    """
    dt = market.T / n_steps
    rng, rng_jump = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    vol_step = np.sqrt(market.diffusion_variance(dt) * dt)

    s = np.full(n_paths, market.s0)
    q = np.zeros(n_paths, dtype=np.int64)
    cash = np.zeros(n_paths)
    n_bid = np.zeros(n_paths, dtype=np.int64)
    n_ask = np.zeros(n_paths, dtype=np.int64)
    spread_sum = 0.0
    hist = {name: [] for name in ("s", "q", "cash", "delta_b", "delta_a", "fill_b", "fill_a")}

    for j in range(n_steps):
        delta_b, delta_a = strategy(market.T - j * dt, q)
        u_b, u_a, z = rng.random(n_paths), rng.random(n_paths), rng.standard_normal(n_paths)
        fill_b = u_b < np.minimum(1.0, market.intensity(delta_b) * dt)
        fill_a = u_a < np.minimum(1.0, market.intensity(delta_a) * dt)
        if q_max is not None:
            fill_b &= q < q_max
            fill_a &= q > -q_max
        if record:
            for name, val in zip(hist, (s, q, cash, delta_b, delta_a, fill_b, fill_a)):
                hist[name].append(np.array(val))
        spread = delta_b + delta_a
        spread_sum += spread[np.isfinite(spread)].mean()

        cash, q = apply_fills(cash, q, s, delta_b, delta_a, fill_b, fill_a)
        n_bid += fill_b
        n_ask += fill_a

        ds = vol_step * z
        if market.impact:
            # every market order moves the mid; those that reach my quote also fill me
            buy_mo = (u_a < market.A * dt) | fill_a
            sell_mo = (u_b < market.A * dt) | fill_b
            ds = ds + market.impact * (buy_mo.astype(float) - sell_mo.astype(float))
        if market.jump_rate:
            jump = rng_jump.random(n_paths) < market.jump_rate * dt
            ds = ds + jump * rng_jump.normal(0.0, 1.0, n_paths) * market.jump_std
        s = s + ds

    path = None
    if record:
        for name, val in zip(("s", "q", "cash"), (s, q, cash)):
            hist[name].append(np.array(val))
        path = {name: np.stack(vals) for name, vals in hist.items()}
        path["t"] = np.arange(n_steps + 1) * dt
    return SimResult(pnl=cash + q * s, inventory=q, bid_fills=n_bid, ask_fills=n_ask,
                     mean_spread=spread_sum / n_steps, path=path)


def certainty_equivalent(pnl: np.ndarray, gamma: float) -> tuple[float, float]:
    """Sample CARA certainty equivalent -(1/gamma) ln mean(exp(-gamma pnl)) and its
    delta-method standard error.

    The estimator is dominated by the worst paths once gamma * std(pnl) exceeds
    about 2; the standard error is then itself unreliable and the estimate is
    biased upwards. Use ``hjb.policy_certainty_equivalent`` for exact values.
    """
    n = pnl.size
    log_m = logsumexp(-gamma * pnl) - np.log(n)
    rel = np.exp(-gamma * pnl - log_m)                     # exp(-gamma W) / mean
    return float(-log_m / gamma), float(rel.std(ddof=1) / (gamma * np.sqrt(n)))


def summarise(res: SimResult, gamma: float) -> dict:
    """The statistics reported in the paper's table, plus the certainty equivalent."""
    ce, ce_se = certainty_equivalent(res.pnl, gamma)
    return {
        "mean_spread": float(res.mean_spread),
        "mean_pnl": float(res.pnl.mean()),
        "std_pnl": float(res.pnl.std(ddof=1)),
        "mean_inventory": float(res.inventory.mean()),
        "std_inventory": float(res.inventory.std(ddof=1)),
        "mean_fills": float((res.bid_fills + res.ask_fills).mean()),
        "ce_mc": ce,
        "ce_mc_se": ce_se,
    }
