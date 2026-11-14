"""Exact solution of the market-making HJB equation with inventory limits, and
exact evaluation of the expected utility of an arbitrary quoting policy.

Value function (Gueant, Lehalle and Fernandez-Tapia 2013). With inventory
q in {-Q..Q}, the substitution

    u(t, s, x, q) = -exp(-gamma (x + q s)) * v_q(t)^(-gamma/k)

turns the HJB equation into the linear system (tau = T - t)

    dv_q/dtau = -alpha q^2 v_q + eta (v_{q-1} + v_{q+1}),     v_q(0) = 1,
    alpha = k gamma sigma^2 / 2,   eta = A (1 + gamma/k)^(-(1 + k/gamma)),

with v_{Q+1} = v_{-Q-1} = 0, i.e. v(tau) = exp(-M tau) 1 for a tridiagonal M.
The optimal quotes are

    delta^b_q = (1/k) ln(v_q / v_{q+1}) + (1/gamma) ln(1 + gamma/k)
    delta^a_q = (1/k) ln(v_q / v_{q-1}) + (1/gamma) ln(1 + gamma/k).

Policy evaluation. For any quotes delta^{b,a}_q(tau) the expected utility has the
same separable form, -exp(-gamma (x + q s)) w_q. For the discretised strategy
that the simulator runs (quotes held over a step of length dt, independent
Bernoulli fills with p = min(1, lambda dt), then a Gaussian move of the mid) w
obeys an exact backward recursion, because E exp(-gamma q' dS) = exp(gamma^2
sigma^2 q'^2 dt / 2):

    w_q <- sum over fills (f_b, f_a) of  P(f_b) P(f_a) exp(-gamma (delta^b f_b + delta^a f_a))
                                         * exp(gamma^2 sigma^2 q'^2 dt / 2) * w_{q'},   q' = q + f_b - f_a,

starting from w = 1 at the horizon. The certainty equivalent from (x, q) = (0, 0)
is -(1/gamma) ln w_0. As dt -> 0 the recursion becomes the linear ODE

    dw_q/dtau = (gamma^2 sigma^2 q^2 / 2 - lambda^b_q - lambda^a_q) w_q
                + lambda^b_q exp(-gamma delta^b_q) w_{q+1} + lambda^a_q exp(-gamma delta^a_q) w_{q-1}.

Numerics. v and w stay positive but their components span hundreds of orders of
magnitude (w_q grows like exp(gamma^2 sigma^2 q^2 tau / 2)), and a general-purpose
matrix exponential loses the small components to rounding. Everything here is
therefore done in log space with sums of non-negative terms only: exp(hB) v is a
Taylor series after shifting the diagonal, and the recursion for w is a logsumexp.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal

_LOG_TOL = np.log(1e-18)


def _log_matvec(log_x, log_diag, log_up, log_down):
    """log(B x) along the last axis for a non-negative tridiagonal B.

    Row q of B has diag[q] on the diagonal, up[q] on x[q+1] and down[q] on x[q-1].
    """
    out = log_diag + log_x
    out[..., :-1] = np.logaddexp(out[..., :-1], log_up[..., :-1] + log_x[..., 1:])
    out[..., 1:] = np.logaddexp(out[..., 1:], log_down[..., 1:] + log_x[..., :-1])
    return out


def log_expm_apply(log_x, diag, up, down, h: float, max_terms: int = 2000):
    """log(exp(h B) x) for tridiagonal B with up, down >= 0 (any sign on the diagonal).

    exp(hB) = exp(-c h) exp(h (B + cI)) with c chosen so that B + cI >= 0 entrywise;
    the Taylor series of the second factor then has no cancellation.
    """
    shift = np.maximum(-np.min(diag, axis=-1, keepdims=True), 0.0)
    with np.errstate(divide="ignore"):
        log_diag, log_up, log_down = np.log(diag + shift), np.log(up), np.log(down)
    term = total = np.asarray(log_x, dtype=float)
    for n in range(1, max_terms):
        term = _log_matvec(term, log_diag, log_up, log_down) + np.log(h / n)
        total = np.logaddexp(total, term)
        if n > 2 and np.all(term < total + _LOG_TOL):
            return total - shift * h
    raise RuntimeError("Taylor series did not converge; reduce the step h")


def glft_coefficients(gamma, sigma, k, A):
    """alpha = k gamma sigma^2 / 2 and eta = A (1 + gamma/k)^(-(1 + k/gamma))."""
    alpha = k * gamma * sigma**2 / 2.0
    eta = A * np.exp(-(1.0 + k / gamma) * np.log1p(gamma / k))
    return alpha, eta


def solve_log_v(gamma, sigma, k, A, Q: int, T: float, n_steps: int) -> np.ndarray:
    """ln v_q(tau_n) on tau_n = n T / n_steps, shape (..., n_steps + 1, 2Q + 1).

    sigma, k, A may be arrays (broadcast together) to solve many parameter sets at once.
    """
    q = np.arange(-Q, Q + 1)
    alpha, eta = glft_coefficients(gamma, sigma, k, A)
    alpha, eta = np.broadcast_arrays(np.asarray(alpha, float), np.asarray(eta, float))
    diag = -alpha[..., None] * q**2
    off = eta[..., None] * np.ones(q.size)
    log_v = [np.zeros(diag.shape)]
    for _ in range(n_steps):
        log_v.append(log_expm_apply(log_v[-1], diag, off, off, T / n_steps))
    return np.stack(log_v, axis=-2)


def quotes_from_log_v(log_v: np.ndarray, gamma: float, k):
    """Optimal (delta_b, delta_a) from ln v; +inf where the inventory limit forbids a quote."""
    k = np.asarray(k, float)[..., None, None] if np.ndim(k) else k
    c = np.log1p(gamma / k) / gamma
    db = np.full(log_v.shape, np.inf)
    da = np.full(log_v.shape, np.inf)
    db[..., :-1] = (log_v[..., :-1] - log_v[..., 1:]) / k + c
    da[..., 1:] = (log_v[..., 1:] - log_v[..., :-1]) / k + c
    return db, da


def exact_quotes(gamma: float, sigma, k, A, Q: int, T: float, n_steps: int):
    """Exact optimal quote tables (delta_b, delta_a), shape (..., n_steps + 1, 2Q + 1)."""
    return quotes_from_log_v(solve_log_v(gamma, sigma, k, A, Q, T, n_steps), gamma, k)


def optimal_certainty_equivalent(gamma: float, sigma: float, k: float, A: float,
                                 Q: int, T: float, n_steps: int = 200) -> float:
    """CE of the optimal policy from (x, q) = (0, 0): u = -v_0^(-gamma/k) => CE = ln v_0(T) / k."""
    return float(solve_log_v(gamma, sigma, k, A, Q, T, n_steps)[-1, Q] / k)


def stationary_quotes(gamma: float, sigma: float, k: float, A: float, Q: int):
    """Exact tau -> infinity limit of the optimal quotes on {-Q..Q}.

    v(tau) ~ exp(-lambda_min tau) f where f > 0 is the ground state of M, so the
    quotes converge to (1/k) ln(f_q / f_{q+-1}) + (1/gamma) ln(1 + gamma/k).
    f comes from a dense eigen-solve, so only components within ~1e-10 of the
    largest are accurate: use the central inventories.
    """
    alpha, eta = glft_coefficients(gamma, sigma, k, A)
    q = np.arange(-Q, Q + 1)
    _, vec = eigh_tridiagonal(alpha * q.astype(float) ** 2, -eta * np.ones(2 * Q),
                              select="i", select_range=(0, 0))
    log_f = np.log(np.abs(vec[:, 0]))
    return quotes_from_log_v(log_f, gamma, k)


def policy_certainty_equivalent(delta_b: np.ndarray, delta_a: np.ndarray, gamma: float,
                                sigma: float, k: float, A: float, T: float) -> np.ndarray:
    """Exact CARA certainty equivalent of a tabulated policy, from (x, q) = (0, 0).

    delta_b, delta_a have shape (..., n_steps + 1, 2Q + 1) with index n <-> tau_n
    = n T / n_steps (see ``quotes.tabulate``). This evaluates, without sampling,
    the Markov chain that ``simulate`` runs with the same n_steps and an inventory
    limit Q, so a simulated certainty equivalent differs from it by Monte Carlo
    noise only. (sigma, k, A) are the *true* parameters of the world; the quotes
    may have been computed from different ones.
    """
    n_steps, n_q = delta_b.shape[-2] - 1, delta_b.shape[-1]
    Q = (n_q - 1) // 2
    dt = T / n_steps
    carry = 0.5 * gamma**2 * sigma**2 * dt * np.arange(-Q, Q + 1) ** 2   # ln E exp(-gamma q dS)
    log_w = np.zeros(delta_b.shape[:-2] + (n_q,))
    with np.errstate(divide="ignore"):
        for n in range(1, n_steps + 1):
            db, da = delta_b[..., n, :], delta_a[..., n, :]
            p_b = np.minimum(1.0, A * np.exp(-k * db) * dt)
            p_a = np.minimum(1.0, A * np.exp(-k * da) * dt)
            cont = carry + log_w                      # by inventory after the fills
            new = cont + np.logaddexp(np.log1p(-p_b) + np.log1p(-p_a),
                                      np.log(p_b) + np.log(p_a) - gamma * (db + da))
            up = np.log(p_b) + np.log1p(-p_a) - gamma * db        # bid fill only: q -> q + 1
            down = np.log(p_a) + np.log1p(-p_b) - gamma * da      # ask fill only: q -> q - 1
            new[..., :-1] = np.logaddexp(new[..., :-1], up[..., :-1] + cont[..., 1:])
            new[..., 1:] = np.logaddexp(new[..., 1:], down[..., 1:] + cont[..., :-1])
            log_w = new
    return -log_w[..., Q] / gamma


class ExactStrategy:
    """The exact optimal quotes as a strategy ``(tau, q) -> (delta_b, delta_a)``.

    The table is solved on n_steps intervals of [0, T]; tau is rounded to the grid,
    so use the simulator's own n_steps. Inventory never leaves [-Q, Q] because the
    quote that would take it outside is +inf.
    """

    def __init__(self, gamma: float, sigma: float, k: float, A: float, Q: int, T: float, n_steps: int):
        self.Q, self.h = Q, T / n_steps
        self.delta_b, self.delta_a = exact_quotes(gamma, sigma, k, A, Q, T, n_steps)

    def __call__(self, tau: float, q: np.ndarray):
        n = int(round(tau / self.h))
        i = np.clip(q + self.Q, 0, 2 * self.Q)
        return self.delta_b[n, i], self.delta_a[n, i]
