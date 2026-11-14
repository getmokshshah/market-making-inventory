"""Closed-form quotes and the simulator: fill rates, accounting, invariants."""
import numpy as np

from mmsim import hjb, quotes
from mmsim.model import Market
from mmsim.simulate import apply_fills, certainty_equivalent, simulate

G, SIG, K, A = 0.1, 2.0, 1.5, 140.0
MKT = Market()


def test_avellaneda_stoikov_identities_and_gamma_to_zero():
    q, tau, s = np.arange(-4, 5), 0.6, 100.0
    db, da = quotes.as_offsets(q, tau, G, SIG, K)
    np.testing.assert_allclose(db + da, quotes.as_spread(tau, G, SIG, K))
    # quotes are centred on the reservation price
    np.testing.assert_allclose(((s + da) + (s - db)) / 2, quotes.as_reservation_price(s, q, tau, G, SIG))
    # gamma -> 0: reservation price -> mid, spread -> 2/k
    np.testing.assert_allclose(quotes.as_reservation_price(s, q, tau, 1e-9, SIG), s, atol=1e-6)
    assert abs(quotes.as_spread(tau, 1e-9, SIG, K) - 2 / K) < 1e-6
    assert abs(quotes.glft_slope(1e-9, SIG, K, A)) < 1e-4


def test_fill_rate_matches_exponential_intensity():
    delta, n_steps = 0.8, 1000
    res = simulate(MKT, quotes.fixed_strategy(delta), 20_000, n_steps, seed=1)
    lam = A * np.exp(-K * delta)
    for fills in (res.bid_fills, res.ask_fills):
        se = np.sqrt(lam * MKT.T / fills.size)
        assert abs(fills.mean() / MKT.T - lam) < 4 * se


def test_wealth_identity_on_scripted_path():
    s = np.array([100.0])
    cash, q = np.array([0.0]), np.array([0])
    script = [  # (delta_b, delta_a, bid fill, ask fill, ds)
        (0.5, 0.7, True, False, 1.0), (0.4, 0.9, True, True, -2.0),
        (0.6, 0.6, False, True, 0.5), (0.3, 0.8, False, False, 3.0)]
    for db, da, fb, fa, ds in script:
        w0 = cash + q * s
        cash, q = apply_fills(cash, q, s, db, da, np.array([fb]), np.array([fa]))
        s = s + ds
        np.testing.assert_allclose(cash + q * s - w0, db * fb + da * fa + q * ds)
    assert q[0] == 0 and np.isclose(cash[0] + q[0] * s[0], 0.5 + 1.0 + 0.4 + 0.9 - 2.0 + 0.6)


def test_wealth_identity_on_simulated_paths():
    res = simulate(MKT, quotes.inventory_strategy(G, SIG, K), 50, 200, seed=3, record=True)
    p = res.path
    w = p["cash"] + p["q"] * p["s"]
    dw = p["delta_b"] * p["fill_b"] + p["delta_a"] * p["fill_a"] + p["q"][1:] * np.diff(p["s"], axis=0)
    np.testing.assert_allclose(np.diff(w, axis=0), dw, atol=1e-9)
    np.testing.assert_allclose(res.pnl, w[-1])
    np.testing.assert_array_equal(res.inventory, p["fill_b"].sum(0) - p["fill_a"].sum(0))


def test_symmetric_strategy_has_zero_mean_and_wider_inventory():
    sym = simulate(MKT, quotes.symmetric_strategy(G, SIG, K), 20_000, 200, seed=5)
    inv = simulate(MKT, quotes.inventory_strategy(G, SIG, K), 20_000, 200, seed=5)
    assert abs(sym.inventory.mean()) < 4 * sym.inventory.std() / np.sqrt(20_000)
    assert sym.inventory.var() > 4 * inv.inventory.var()
    assert sym.pnl.std() > inv.pnl.std()


def test_seeded_runs_reproduce():
    strat = quotes.inventory_strategy(G, SIG, K)
    a, b, c = (simulate(MKT, strat, 500, 100, seed=s) for s in (11, 11, 12))
    np.testing.assert_array_equal(a.pnl, b.pnl)
    np.testing.assert_array_equal(a.inventory, b.inventory)
    assert not np.array_equal(a.pnl, c.pnl)


def test_adverse_selection_and_jumps_off_reproduce_baseline():
    strat = quotes.inventory_strategy(G, SIG, K)
    base = simulate(MKT, strat, 2000, 200, seed=7)
    for mkt in (Market(impact=0.0), Market(jump_rate=3.0, jump_std=0.0), Market(jump_rate=0.0, jump_std=1.0)):
        other = simulate(mkt, strat, 2000, 200, seed=7)
        np.testing.assert_array_equal(base.pnl, other.pnl)
        np.testing.assert_array_equal(base.inventory, other.inventory)
    assert not np.array_equal(base.pnl, simulate(Market(impact=0.05), strat, 2000, 200, seed=7).pnl)


def test_impact_moves_mid_against_fills_and_preserves_variance():
    xi, n_steps = 0.08, 1000
    mkt = Market(impact=xi)
    res = simulate(mkt, quotes.fixed_strategy(1 / K), 4000, n_steps, seed=9, record=True)
    ds = np.diff(res.path["s"], axis=0)
    fa, fb = res.path["fill_a"], res.path["fill_b"]
    expected = xi * (1 - A * mkt.T / n_steps)           # the opposite side may also trade in the step
    assert abs(ds[fa].mean() - expected) < 4 * ds[fa].std() / np.sqrt(fa.sum())
    assert abs(ds[fb].mean() + expected) < 4 * ds[fb].std() / np.sqrt(fb.sum())
    # total variance of the mid stays sigma^2 T
    var_T = res.path["s"][-1].var()
    assert abs(var_T / (SIG**2 * mkt.T) - 1) < 4 * np.sqrt(2 / 4000)
    jump = simulate(Market(jump_rate=0.5, jump_std=2.0), quotes.fixed_strategy(1 / K), 4000, 200, seed=10, record=True)
    # jump paths are fat-tailed: Var of the sample variance is (kurtosis - 1)/n, kurtosis = 3 + 3*8/16 = 4.5
    assert abs(jump.path["s"][-1].var() / (SIG**2) - 1) < 4 * np.sqrt(3.5 / 4000)


def test_simulated_certainty_equivalent_matches_exact_recursion():
    Q, n_steps, n_paths = 15, 200, 100_000
    exact = hjb.ExactStrategy(G, SIG, K, A, Q, MKT.T, n_steps)
    cases = [(exact, (exact.delta_b, exact.delta_a)),
             *((s, quotes.tabulate(s, Q, MKT.T, n_steps))
               for s in (quotes.inventory_strategy(G, SIG, K), quotes.fixed_strategy(1 / K)))]
    for strat, tables in cases:
        res = simulate(MKT, strat, n_paths, n_steps, seed=13, q_max=Q)
        assert np.abs(res.inventory).max() <= Q
        ce_mc, se = certainty_equivalent(res.pnl, G)
        ce_exact = hjb.policy_certainty_equivalent(*tables, G, SIG, K, A, MKT.T)
        assert abs(ce_mc - ce_exact) < 4 * se             # same Markov chain: Monte Carlo noise only
