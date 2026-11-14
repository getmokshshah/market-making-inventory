"""The exact solution: the ODE system, its limits, and the policy evaluator."""
import numpy as np
from scipy.linalg import expm

from mmsim import hjb, quotes

G, SIG, K, A = 0.1, 2.0, 1.5, 140.0


def test_log_expm_apply_matches_scipy_expm():
    rng = np.random.default_rng(0)
    n = 9
    diag, up, down = rng.normal(0, 3, n), rng.uniform(0, 4, n), rng.uniform(0, 4, n)
    B = np.diag(diag) + np.diag(up[:-1], 1) + np.diag(down[1:], -1)
    x = rng.uniform(0.1, 5, n)
    got = np.exp(hjb.log_expm_apply(np.log(x), diag, up, down, 0.7))
    np.testing.assert_allclose(got, expm(0.7 * B) @ x, rtol=1e-11)


def test_v_is_matrix_exponential_and_terminal_condition():
    Q, T = 8, 0.5
    log_v = hjb.solve_log_v(G, SIG, K, A, Q, T, 50)
    assert np.all(log_v[0] == 0.0)                      # v_q(T) = 1
    alpha, eta = hjb.glft_coefficients(G, SIG, K, A)
    q = np.arange(-Q, Q + 1)
    M = np.diag(alpha * q**2) - eta * (np.eye(2 * Q + 1, k=1) + np.eye(2 * Q + 1, k=-1))
    np.testing.assert_allclose(np.exp(log_v[-1]), expm(-M * T) @ np.ones(2 * Q + 1), rtol=1e-9)


def test_ode_residual_by_finite_differences():
    Q, T, n = 10, 0.02, 2000
    h = T / n
    v = np.exp(hjb.solve_log_v(G, SIG, K, A, Q, T, n))
    alpha, eta = hjb.glft_coefficients(G, SIG, K, A)
    q = np.arange(-Q, Q + 1)
    vpad = np.pad(v, ((0, 0), (1, 1)))                  # v = 0 outside the inventory limits
    rhs = -alpha * q**2 * v + eta * (vpad[:, :-2] + vpad[:, 2:])
    dv = (v[2:] - v[:-2]) / (2 * h)
    assert np.max(np.abs(dv - rhs[1:-1])) / np.max(np.abs(rhs)) < 1e-6


def test_exact_quotes_converge_to_stationary_and_glft_closed_form():
    Q = 30
    db, da = hjb.exact_quotes(G, SIG, K, A, Q, 3.0, 300)
    sb, sa = hjb.stationary_quotes(G, SIG, K, A, Q)
    mid = slice(Q - 10, Q + 11)
    np.testing.assert_allclose(db[-1, mid], sb[mid], atol=1e-8)
    np.testing.assert_allclose(da[-1, mid], sa[mid], atol=1e-8)
    # the closed form replaces the discrete ground state by a Gaussian: close, not exact
    q = np.arange(-5, 6)
    gb, ga = quotes.glft_asymptotic_offsets(q, G, SIG, K, A)
    np.testing.assert_allclose(db[-1, Q - 5:Q + 6], gb, atol=5e-3)
    np.testing.assert_allclose(da[-1, Q - 5:Q + 6], ga, atol=5e-3)


def test_exact_quotes_match_avellaneda_stoikov_near_horizon():
    Q, T = 30, 0.01
    db, da = hjb.exact_quotes(G, SIG, K, A, Q, T, 2)    # tau = 0, 0.005, 0.01
    q = np.arange(-3, 4)
    err = []
    for n, tau in ((1, 0.005), (2, 0.01)):
        ab, aa = quotes.as_offsets(q, tau, G, SIG, K)
        err.append(max(np.abs(db[n, Q - 3:Q + 4] - ab).max(), np.abs(da[n, Q - 3:Q + 4] - aa).max()))
    skew = G * SIG**2 * 0.01                            # size of the leading-order term
    assert err[1] < 0.02 * skew
    assert err[1] / err[0] > 3.5                        # error is at least second order in tau


def test_gamma_to_zero_exact_quotes_tend_to_one_over_k():
    # short horizon so the inventory limit, 30 units away, cannot be felt at q near 0
    Q = 30
    db, da = hjb.exact_quotes(1e-6, SIG, K, A, Q, 0.1, 10)
    np.testing.assert_allclose(db[-1, Q - 5:Q + 6], 1 / K, atol=1e-4)
    np.testing.assert_allclose(da[-1, Q - 5:Q + 6], 1 / K, atol=1e-4)


def test_policy_evaluator_reproduces_optimal_value_and_optimum_dominates():
    # the discrete-time evaluation of the optimal quotes converges to the continuous-time value ln v_0 / k
    Q, T, n = 15, 1.0, 2000
    ce_opt = hjb.optimal_certainty_equivalent(G, SIG, K, A, Q, T, 200)
    gap = [abs(hjb.policy_certainty_equivalent(*hjb.exact_quotes(G, SIG, K, A, Q, T, m), G, SIG, K, A, T) - ce_opt)
           for m in (500, n)]
    assert gap[1] < 0.002 * ce_opt and gap[1] < gap[0] / 2
    ce_tab = hjb.policy_certainty_equivalent(*hjb.exact_quotes(G, SIG, K, A, Q, T, n), G, SIG, K, A, T)
    for strat in (quotes.inventory_strategy(G, SIG, K), quotes.symmetric_strategy(G, SIG, K),
                  quotes.fixed_strategy(1 / K)):
        ce = hjb.policy_certainty_equivalent(*quotes.tabulate(strat, Q, T, n), G, SIG, K, A, T)
        assert ce < ce_tab


def test_batched_solve_matches_single():
    sig = np.array([1.0, 2.0, 3.0])
    batch = hjb.solve_log_v(G, sig, K, A, 6, 0.3, 20)
    for i, s in enumerate(sig):
        np.testing.assert_allclose(batch[i], hjb.solve_log_v(G, s, K, A, 6, 0.3, 20), rtol=1e-12, atol=1e-12)
