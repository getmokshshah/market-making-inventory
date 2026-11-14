"""Run every computation behind the README: exact quotes, the replication of the
paper's simulation, and the experiments that leave the model.

Writes results/summary.json, results/experiments.json and results/web/*.json.
All choices (parameters, grids, seeds) are fixed in this file and in mmsim.model.
"""
from __future__ import annotations

import time

import numpy as np

from mmsim import hjb, quotes
from mmsim.model import DT_PAPER, GAMMA, GAMMAS_PAPER, PAPER, Q_MAX, SEED, Market
from mmsim.results import update_summary, write_json
from mmsim.simulate import simulate, summarise

M = PAPER
N_PAPER = round(M.T / DT_PAPER)          # 200 steps, the paper's dt = 0.005
N_FINE = 1000                            # dt = 0.001 for the experiments
PATHS_PAPER, PATHS_FINE, PATHS_SWEEP = 100_000, 20_000, 10_000
FRONTIER_GAMMAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
IMPACTS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
WIDENINGS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
MULTIPLIERS = 2.0 ** (np.arange(-3, 4) / 3)          # 0.5 ... 2, log-spaced
STRATEGIES = ("inventory", "exact", "symmetric", "fixed")


def make_strategy(name: str, gamma: float, n_steps: int):
    if name == "inventory":
        return quotes.inventory_strategy(gamma, M.sigma, M.k)
    if name == "symmetric":
        return quotes.symmetric_strategy(gamma, M.sigma, M.k)
    if name == "fixed":
        return quotes.fixed_strategy(1.0 / M.k)
    return hjb.ExactStrategy(gamma, M.sigma, M.k, M.A, Q_MAX, M.T, n_steps)


def ce_exact(strategy, gamma: float, n_steps: int) -> float:
    """Exact certainty equivalent under the inventory limit Q_MAX."""
    if isinstance(strategy, hjb.ExactStrategy):
        tables = strategy.delta_b, strategy.delta_a
    else:
        tables = quotes.tabulate(strategy, Q_MAX, M.T, n_steps)
    return float(hjb.policy_certainty_equivalent(*tables, gamma, M.sigma, M.k, M.A, M.T))


def run(name: str, gamma: float, n_steps: int, n_paths: int, market: Market = M,
        q_max: int | None = Q_MAX, widen: float = 0.0) -> dict:
    strategy = make_strategy(name, gamma, n_steps)
    sim_strategy = quotes.widened(strategy, widen) if widen else strategy
    res = simulate(market, sim_strategy, n_paths, n_steps, SEED, q_max=q_max)
    row = {"strategy": name, "gamma": gamma, **summarise(res, gamma)}
    row["mean_pnl_se"] = row["std_pnl"] / np.sqrt(n_paths)
    if market == M and not widen:
        row["ce_exact"] = ce_exact(strategy, gamma, n_steps)
    return row


def quotes_section() -> dict:
    """Exact quotes against the AS approximation and the GLFT closed form."""
    g, Q = GAMMA, Q_MAX
    q = np.arange(-Q, Q + 1)
    db, da = hjb.exact_quotes(g, M.sigma, M.k, M.A, Q, M.T, N_PAPER)
    gb, ga = quotes.glft_asymptotic_offsets(q, g, M.sigma, M.k, M.A)
    tau_grid = np.arange(N_PAPER + 1) * M.T / N_PAPER
    by_tau = []
    for tau in (0.05, 0.25, 0.5, 1.0):
        n = round(tau / M.T * N_PAPER)
        ab, aa = quotes.as_offsets(q, tau, g, M.sigma, M.k)
        by_tau.append({"tau": tau, "exact_bid": db[n], "exact_ask": da[n], "as_bid": ab, "as_ask": aa})
    params = {"s0": M.s0, "T": M.T, "sigma": M.sigma, "A": M.A, "k": M.k, "gamma": g, "Q": Q}
    write_json("web/quotes_by_inventory.json", {
        "description": "Offsets from the mid: bid = s - bid_offset, ask = s + ask_offset. "
                       "null = no quote (inventory limit). tau = T - t.",
        "params": params, "q": q, "base_half_spread": quotes.base_half_spread(g, M.k),
        "glft_asymptotic_bid": gb, "glft_asymptotic_ask": ga, "by_tau": by_tau})

    # skew per unit of inventory: how far the quote midpoint sits below the mid at q = 1
    skew_exact = 0.5 * (db[:, Q + 1] - da[:, Q + 1])
    skew_as = g * M.sigma**2 * tau_grid
    slope = quotes.glft_slope(g, M.sigma, M.k, M.A)
    sb, _ = hjb.stationary_quotes(g, M.sigma, M.k, M.A, Q)
    c5, c10 = slice(Q - 5, Q + 6), slice(Q - 10, Q + 11)
    n05 = round(0.05 / M.T * N_PAPER)
    ab05, _ = quotes.as_offsets(q, 0.05, g, M.sigma, M.k)
    ab1, _ = quotes.as_offsets(q, 1.0, g, M.sigma, M.k)
    summary = {
        "base_half_spread": quotes.base_half_spread(g, M.k),
        "skew_per_unit_as_tau1": float(skew_as[-1]),
        "skew_per_unit_exact_tau1": float(skew_exact[-1]),
        "skew_per_unit_glft_closed_form": slope,
        "as_over_exact_skew_tau1": float(skew_as[-1] / skew_exact[-1]),
        "tau_exact_skew_reaches_95pct_of_limit": float(tau_grid[np.argmax(skew_exact >= 0.95 * skew_exact[-1])]),
        "max_abs_exact_minus_as_bid_tau0.05_absq_le5": float(np.abs(db[n05, c5] - ab05[c5]).max()),
        "max_abs_exact_minus_as_bid_tau1_absq_le5": float(np.abs(db[-1, c5] - ab1[c5]).max()),
        "max_abs_exact_minus_glft_bid_tau1_absq_le10": float(np.abs(db[-1, c10] - gb[c10]).max()),
        "max_abs_exact_minus_stationary_bid_tau1_absq_le10": float(np.abs(db[-1, c10] - sb[c10]).max()),
        "optimal_ce_theory": hjb.optimal_certainty_equivalent(g, M.sigma, M.k, M.A, Q, M.T, N_FINE),
    }
    curves = {"q": q, "tau": tau_grid, "skew_exact": skew_exact, "skew_as": skew_as, "skew_glft": slope,
              "exact_ask_tau1": da[-1], "as_ask_tau1": quotes.as_offsets(q, 1.0, g, M.sigma, M.k)[1],
              "glft_ask": ga, "exact_ask_tau005": da[n05], "as_ask_tau005": quotes.as_offsets(q, 0.05, g, M.sigma, M.k)[1]}
    return summary, curves


def replication_section() -> dict:
    """The paper's simulation: dt = 0.005, no inventory limit except in the exact strategy."""
    rows = [run(name, g, N_PAPER, PATHS_PAPER, q_max=None) for g in GAMMAS_PAPER for name in STRATEGIES]
    write_json("web/strategy_table.json", {
        "description": "Monte Carlo statistics of terminal P&L and inventory. ce_mc is the sample CARA "
                       "certainty equivalent (unreliable when gamma*std_pnl > 2); ce_exact is the exact value "
                       "of the same quoting rule under an inventory limit of q_max_exact.",
        "params": {"s0": M.s0, "T": M.T, "sigma": M.sigma, "A": M.A, "k": M.k, "q0": 0, "dt": DT_PAPER,
                   "n_steps": N_PAPER, "n_paths": PATHS_PAPER, "seed": SEED, "q_max_exact": Q_MAX,
                   "fixed_half_spread": 1.0 / M.k, "rng": "numpy default_rng (PCG64)"},
        "discretisation": [
            "each step: quotes from (T - t, q) before any fill",
            "at most one bid fill and one ask fill per step, independent, each with probability min(1, A*exp(-k*delta)*dt)",
            "bid fill: q += 1, cash -= s - delta_b; ask fill: q -= 1, cash += s + delta_a",
            "then the mid moves by sigma*sqrt(dt)*N(0,1)",
            "inventory and symmetric: spread = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k), centred on "
            "r = s - q*gamma*sigma^2*(T-t) and on s respectively; no inventory limit",
            "exact: GLFT optimal quotes with limit q_max_exact, table step = dt; fixed: half-spread 1/k",
            "pnl = cash_T + q_T * s_T; mean_spread = average of delta_a + delta_b over steps and paths"],
        "rows": rows})
    fine = {name: run(name, GAMMA, 2000, PATHS_FINE, q_max=None) for name in ("inventory", "symmetric")}
    capped = run("symmetric", GAMMA, N_PAPER, PATHS_PAPER, q_max=Q_MAX)
    keep = ("mean_spread", "mean_pnl", "std_pnl", "mean_inventory", "std_inventory", "mean_fills", "ce_mc", "ce_exact")
    table = {f"gamma_{r['gamma']}": {} for r in rows}
    for r in rows:
        table[f"gamma_{r['gamma']}"][r["strategy"]] = {key: r[key] for key in keep}
    by = table[f"gamma_{GAMMA}"]
    return {
        "n_paths": PATHS_PAPER, "dt": DT_PAPER, "seed": SEED, "table": table,
        "gamma_0.1_pnl_std_ratio_symmetric_over_inventory": by["symmetric"]["std_pnl"] / by["inventory"]["std_pnl"],
        "gamma_0.1_inventory_std_ratio_symmetric_over_inventory":
            by["symmetric"]["std_inventory"] / by["inventory"]["std_inventory"],
        "gamma_0.1_mean_pnl_given_up_pct": 100 * (1 - by["inventory"]["mean_pnl"] / by["symmetric"]["mean_pnl"]),
        "mean_pnl_se_max": max(r["mean_pnl_se"] for r in rows),
        "dt_check_dt0.0005": {name: {key: fine[name][key] for key in ("mean_pnl", "std_pnl", "std_inventory")}
                              for name in fine},
        "symmetric_gamma_0.1_with_limit_30": {key: capped[key] for key in ("mean_pnl", "std_pnl", "std_inventory")},
    }


def frontier_section():
    rows = [run(name, g, N_FINE, PATHS_FINE) for g in FRONTIER_GAMMAS for name in STRATEGIES]
    at = {(r["strategy"], r["gamma"]): r for r in rows}
    summary = {"n_paths": PATHS_FINE, "dt": M.T / N_FINE, "q_max": Q_MAX, "ce_exact": {}, "ce_mc": {}}
    for g in FRONTIER_GAMMAS:
        summary["ce_exact"][f"gamma_{g}"] = {name: at[name, g]["ce_exact"] for name in STRATEGIES}
        summary["ce_mc"][f"gamma_{g}"] = {name: at[name, g]["ce_mc"] for name in STRATEGIES}
    b = {name: at[name, GAMMA] for name in STRATEGIES}
    summary["gamma_0.1"] = {name: {key: b[name][key] for key in ("mean_pnl", "std_pnl", "std_inventory", "ce_mc",
                                                                   "ce_mc_se", "ce_exact")} for name in STRATEGIES}
    summary["gamma_0.1_ce_gain_exact_over_inventory"] = b["exact"]["ce_exact"] - b["inventory"]["ce_exact"]
    summary["exact_has_highest_ce_at_every_gamma"] = all(
        at["exact", g]["ce_exact"] >= max(at[name, g]["ce_exact"] for name in STRATEGIES) for g in FRONTIER_GAMMAS)
    return summary, rows


def adverse_selection_section():
    rows = []
    for xi in IMPACTS:
        market = Market(impact=xi)
        for name in STRATEGIES:
            for widen in ((0.0,) if xi == 0 else (0.0, xi)):
                row = run(name, GAMMA, N_FINE, PATHS_SWEEP, market=market, widen=widen)
                rows.append({"impact": xi, "widen": widen, **row})
    xi_top = IMPACTS[-1]
    sweep = [{"impact": xi_top, "widen": w,
              **run("exact", GAMMA, N_FINE, PATHS_SWEEP, market=Market(impact=xi_top), widen=w)} for w in WIDENINGS]
    at = {(r["strategy"], r["impact"], r["widen"]): r for r in rows}
    summary = {"n_paths": PATHS_SWEEP, "dt": M.T / N_FINE, "impact_bound": M.sigma / np.sqrt(2 * M.A),
               "impact_top": xi_top, "by_strategy": {}}
    for name in STRATEGIES:
        base, hit, fixed = at[name, 0.0, 0.0], at[name, xi_top, 0.0], at[name, xi_top, xi_top]
        summary["by_strategy"][name] = {
            "mean_pnl_no_impact": base["mean_pnl"], "mean_pnl_impact": hit["mean_pnl"],
            "mean_pnl_impact_widened": fixed["mean_pnl"],
            "loss_pct": 100 * (1 - hit["mean_pnl"] / base["mean_pnl"]),
            # each fill loses impact * (1 - A dt): the other side may also trade in the same step
            "first_order_loss_prediction": xi_top * (1 - M.A * M.T / N_FINE) * base["mean_fills"],
            "loss": base["mean_pnl"] - hit["mean_pnl"],
            "recovered_by_widening": fixed["mean_pnl"] - hit["mean_pnl"],
            "recovered_pct_of_loss": 100 * (fixed["mean_pnl"] - hit["mean_pnl"]) / (base["mean_pnl"] - hit["mean_pnl"]),
            "ce_mc_impact": hit["ce_mc"], "ce_mc_impact_widened": fixed["ce_mc"]}
    best = max(sweep, key=lambda r: r["mean_pnl"])
    summary["widening_sweep_exact"] = {f"w_{r['widen']}": r["mean_pnl"] for r in sweep}
    summary["widening_sweep_best_w"] = best["widen"]
    summary["mean_pnl_se_max"] = max(r["mean_pnl_se"] for r in rows)
    return summary, {"rows": rows, "sweep": sweep}


def misspecification_section():
    """Exact strategy computed from (m_sigma sigma, m_k k), evaluated in the true world."""
    g, m = GAMMA, MULTIPLIERS
    db, da = hjb.exact_quotes(g, M.sigma * m[:, None], M.k * m[None, :], M.A, Q_MAX, M.T, N_PAPER)
    ce = hjb.policy_certainty_equivalent(db, da, g, M.sigma, M.k, M.A, M.T)   # [sigma multiplier, k multiplier]
    mid = len(m) // 2
    loss = ce[mid, mid] - ce
    # Monte Carlo spot check of the worst corner
    i, j = np.unravel_index(np.argmax(loss), loss.shape)
    wrong = hjb.ExactStrategy(g, M.sigma * m[i], M.k * m[j], M.A, Q_MAX, M.T, N_PAPER)
    mc = summarise(simulate(M, wrong, PATHS_PAPER, N_PAPER, SEED), g)
    summary = {
        "ce_correct": float(ce[mid, mid]),
        "max_loss_sigma_only": float(loss[:, mid].max()), "max_loss_k_only": float(loss[mid, :].max()),
        "loss_sigma_x0.5": float(loss[0, mid]), "loss_sigma_x2": float(loss[-1, mid]),
        "loss_k_x0.5": float(loss[mid, 0]), "loss_k_x2": float(loss[mid, -1]),
        "n_steps": N_PAPER, "max_loss": float(loss.max()),
        "k_over_sigma_loss_ratio": float(loss[mid, :].max() / loss[:, mid].max()), "max_loss_pct_of_ce": float(100 * loss.max() / ce[mid, mid]),
        "worst_cell": {"sigma_multiplier": float(m[i]), "k_multiplier": float(m[j]),
                       "ce_exact": float(ce[i, j]), "ce_mc": mc["ce_mc"], "ce_mc_se": mc["ce_mc_se"]},
        "min_loss": float(loss.min()),
    }
    return summary, {"multipliers": m, "ce": ce, "loss": loss}


def main() -> None:
    t0 = time.time()
    update_summary("params", {"s0": M.s0, "T": M.T, "sigma": M.sigma, "A": M.A, "k": M.k, "gamma": GAMMA,
                              "q_max": Q_MAX, "seed": SEED, "dt_paper": DT_PAPER, "dt_experiments": M.T / N_FINE})
    experiments = {}
    for name, section in (("quotes", quotes_section), ("frontier", frontier_section),
                          ("adverse_selection", adverse_selection_section),
                          ("misspecification", misspecification_section)):
        summary, experiments[name] = section()
        update_summary(name, summary)
        print(f"{name:20s} done at {time.time() - t0:6.1f}s")
    update_summary("replication", replication_section())
    write_json("experiments.json", experiments)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
