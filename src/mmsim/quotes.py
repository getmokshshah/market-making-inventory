"""Closed-form quotes and the quoting strategies.

Notation: s mid, q inventory, tau = T - t time to horizon, delta^b / delta^a the
distances of the bid / ask from the mid (bid = s - delta^b, ask = s + delta^a).

Avellaneda and Stoikov (2008), approximate solution of the HJB equation:

    reservation price   r = s - q gamma sigma^2 tau
    spread              delta^a + delta^b = gamma sigma^2 tau + (2/gamma) ln(1 + gamma/k)
    =>  delta^b = (1 + 2q) gamma sigma^2 tau / 2 + (1/gamma) ln(1 + gamma/k)
        delta^a = (1 - 2q) gamma sigma^2 tau / 2 + (1/gamma) ln(1 + gamma/k)

Gueant, Lehalle and Fernandez-Tapia (2013), closed-form asymptotic (tau -> infinity):

    delta^b = (1/gamma) ln(1 + gamma/k) + (2q + 1)/2 * sqrt(sigma^2 gamma / (2 k A) * (1 + gamma/k)^(1 + k/gamma))
    delta^a = (1/gamma) ln(1 + gamma/k) - (2q - 1)/2 * sqrt(  same  )

A strategy is a callable ``(tau, q) -> (delta_b, delta_a)`` vectorised over q.
An offset of +inf means "no quote on that side".
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Strategy = Callable[[float, np.ndarray], tuple[np.ndarray, np.ndarray]]


def base_half_spread(gamma: float, k: float) -> float:
    """(1/gamma) ln(1 + gamma/k); tends to 1/k as gamma -> 0."""
    return float(np.log1p(gamma / k) / gamma)


def as_reservation_price(s, q, tau: float, gamma: float, sigma: float):
    """r = s - q gamma sigma^2 tau."""
    return s - q * gamma * sigma**2 * tau


def as_spread(tau, gamma: float, sigma: float, k: float):
    """delta^a + delta^b = gamma sigma^2 tau + (2/gamma) ln(1 + gamma/k)."""
    return gamma * sigma**2 * tau + 2.0 * base_half_spread(gamma, k)


def as_offsets(q, tau: float, gamma: float, sigma: float, k: float):
    """Avellaneda-Stoikov quotes: half the spread either side of the reservation price."""
    q = np.asarray(q, dtype=float)
    half = 0.5 * as_spread(tau, gamma, sigma, k)
    skew = q * gamma * sigma**2 * tau          # s - r
    return half + skew, half - skew


def glft_slope(gamma: float, sigma: float, k: float, A: float) -> float:
    """Asymptotic skew per unit of inventory, sqrt(sigma^2 gamma/(2kA) (1+gamma/k)^(1+k/gamma))."""
    return float(np.sqrt(sigma**2 * gamma / (2.0 * k * A) * (1.0 + gamma / k) ** (1.0 + k / gamma)))


def glft_asymptotic_offsets(q, gamma: float, sigma: float, k: float, A: float):
    """GLFT closed-form quotes far from the horizon (independent of tau)."""
    q = np.asarray(q, dtype=float)
    c, slope = base_half_spread(gamma, k), glft_slope(gamma, sigma, k, A)
    return c + (2 * q + 1) / 2 * slope, c - (2 * q - 1) / 2 * slope


# --- strategies ---------------------------------------------------------------

def inventory_strategy(gamma: float, sigma: float, k: float) -> Strategy:
    """Avellaneda-Stoikov: the AS spread centred on the reservation price."""
    def quotes(tau: float, q: np.ndarray):
        return as_offsets(q, tau, gamma, sigma, k)
    return quotes


def symmetric_strategy(gamma: float, sigma: float, k: float) -> Strategy:
    """The paper's benchmark: the same AS spread, centred on the mid."""
    def quotes(tau: float, q: np.ndarray):
        half = np.full(np.shape(q), 0.5 * as_spread(tau, gamma, sigma, k))
        return half, half
    return quotes


def fixed_strategy(half_spread: float) -> Strategy:
    """Constant quotes at s -/+ half_spread. 1/k maximises delta * A exp(-k delta)."""
    def quotes(tau: float, q: np.ndarray):
        half = np.full(np.shape(q), float(half_spread))
        return half, half
    return quotes


def widened(strategy: Strategy, w: float) -> Strategy:
    """The same strategy with both quotes moved w further from the mid."""
    def quotes(tau: float, q: np.ndarray):
        db, da = strategy(tau, q)
        return db + w, da + w
    return quotes


def tabulate(strategy: Strategy, Q: int, T: float, n_steps: int):
    """Quote tables on the grid tau_n = n T / n_steps, q = -Q..Q, with the inventory
    limit imposed: no bid at q = Q and no ask at q = -Q.

    Returns (delta_b, delta_a), each of shape (n_steps + 1, 2Q + 1).
    """
    q = np.arange(-Q, Q + 1)
    db = np.empty((n_steps + 1, 2 * Q + 1))
    da = np.empty_like(db)
    for n in range(n_steps + 1):
        db[n], da[n] = strategy(n * T / n_steps, q)
    db[:, -1] = np.inf
    da[:, 0] = np.inf
    return db, da
