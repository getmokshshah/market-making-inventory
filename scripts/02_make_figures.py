"""Draw the six figures in figures/ from results/*.json (run 01_run_experiments.py first).

The sample path and the two terminal distributions are re-simulated here with the
replication settings and seed, since the raw paths are not stored.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mmsim import quotes
from mmsim.model import DT_PAPER, GAMMA, PAPER, SEED
from mmsim.plotstyle import (AQUA, BLUE, FIGSIZE, FIGSIZE_TALL, FIGSIZE_WIDE, INK, INK_2, MAGENTA, MUTED, ORANGE,
                             SEQ_CMAP, YELLOW, apply_style, legend, save, title)
from mmsim.results import read_json
from mmsim.simulate import simulate

M = PAPER
N = round(M.T / DT_PAPER)
COLOUR = {"inventory": BLUE, "symmetric": ORANGE, "exact": AQUA, "fixed": YELLOW}
LABEL = {"inventory": "Inventory (Avellaneda-Stoikov)", "symmetric": "Symmetric", "exact": "Exact optimum",
         "fixed": "Fixed spread 2/k"}


def sample_path() -> None:
    res = simulate(M, quotes.inventory_strategy(GAMMA, M.sigma, M.k), 1, N, SEED, record=True)
    p = {key: val[:, 0] for key, val in res.path.items() if key != "t"}
    t = res.path["t"]
    s, q = p["s"][:-1], p["q"][:-1]
    r = quotes.as_reservation_price(s, q, M.T - t[:-1], GAMMA, M.sigma)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_TALL, sharex=True, height_ratios=[2.2, 1])
    ax.plot(t[:-1], s + p["delta_a"], color=MUTED, lw=0.9, label="Bid and ask quotes")
    ax.plot(t[:-1], s - p["delta_b"], color=MUTED, lw=0.9)
    ax.plot(t[:-1], s, color=INK, lw=1.2, label="Mid price")
    ax.plot(t[:-1], r, color=BLUE, lw=1.4, label="Reservation price")
    ax.set_ylabel("Price")
    legend(ax, loc="best", ncols=3)
    title(ax, "Quotes are centred on the reservation price, not the mid",
          "One simulated session, inventory strategy, $\\gamma$ = 0.1. The reservation price sits below the mid when long.")
    ax2.step(t, p["q"], where="post", color=BLUE, lw=1.4)
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.set_ylabel("Inventory (units)")
    ax2.set_xlabel("Time t  (horizon T = 1)")
    title(ax2, "Inventory stays within a few units; the pull toward zero weakens as t approaches T")
    fig.tight_layout()
    save(fig, "sample-path")


def distributions(rep: dict) -> None:
    runs = {name: simulate(M, make(GAMMA, M.sigma, M.k), rep["n_paths"], N, SEED)
            for name, make in (("inventory", quotes.inventory_strategy), ("symmetric", quotes.symmetric_strategy))}
    tab = rep["table"][f"gamma_{GAMMA}"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    bins = np.linspace(10, 125, 93)
    for name, res in runs.items():
        ax.hist(res.pnl, bins=bins, density=True, histtype="stepfilled", alpha=0.55, color=COLOUR[name],
                label=f"{LABEL[name].split(' (')[0]}: mean {tab[name]['mean_pnl']:.1f}, sd {tab[name]['std_pnl']:.1f}")
    ax.set_xlabel("Terminal P&L")
    ax.set_ylabel("Density")
    ax.set_xlim(10, 125)
    legend(ax, loc="upper right")
    title(ax, "Half the P&L risk for 5% of the mean",
          f"Terminal P&L, {rep['n_paths']:,} paths, same spread in both strategies.")
    for name, res in runs.items():
        vals, counts = np.unique(res.inventory, return_counts=True)
        ax2.step(vals, counts / res.inventory.size, where="mid", color=COLOUR[name], lw=1.6,
                 label=f"{LABEL[name].split(' (')[0]}: sd {tab[name]['std_inventory']:.1f}")
    ax2.set_xlabel("Terminal inventory (units)")
    ax2.set_ylabel("Probability")
    ax2.set_xlim(-30, 30)
    legend(ax2, loc="upper left")
    title(ax2, "Terminal inventory is about 3x tighter", "Distribution of $q_T$ over the same paths.")
    fig.tight_layout()
    save(fig, "pnl-and-inventory")


def quotes_figure(c: dict, s: dict) -> None:
    q, tau = np.array(c["q"]), np.array(c["tau"])
    keep = np.abs(q) <= 8
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    ax.plot(q[keep], np.array(c["as_ask_tau1"])[keep], color=BLUE, label="Avellaneda-Stoikov approximation")
    ax.plot(q[keep], np.array(c["glft_ask"])[keep], color=MAGENTA, ls="--", lw=1.6, label="GLFT closed form")
    ax.plot(q[keep], np.array(c["exact_ask_tau1"], dtype=float)[keep], "o", color=AQUA, ms=4.5, label="Exact (ODE system)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Inventory q")
    ax.set_ylabel("Ask offset $\\delta^a$ from the mid")
    legend(ax, loc="lower left")
    title(ax, f"Far from the horizon AS skews {s['as_over_exact_skew_tau1']:.1f}x too hard",
          "Optimal ask offset at T - t = 1. The bid is the mirror image, $\\delta^b(q)=\\delta^a(-q)$.")
    ax2.plot(tau, c["skew_as"], color=BLUE, label="Avellaneda-Stoikov: $\\gamma\\sigma^2(T-t)$")
    ax2.axhline(c["skew_glft"], color=MAGENTA, ls="--", lw=1.6, label="GLFT closed form")
    ax2.plot(tau, c["skew_exact"], color=AQUA, label="Exact")
    ax2.set_ylim(0, 0.12)
    ax2.set_xlabel("Time to horizon T - t")
    ax2.set_ylabel("Quote shift per unit of inventory")
    legend(ax2, loc="upper right")
    title(ax2, "The exact skew follows AS near T, then saturates",
          "Distance of the quote midpoint from the mid at q = 1 (AS line leaves the panel).")
    fig.tight_layout()
    save(fig, "quotes-by-inventory")


def frontier(rows: list, s: dict) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    gammas = sorted({r["gamma"] for r in rows})
    for name in ("inventory", "exact", "symmetric"):
        sel = sorted((r for r in rows if r["strategy"] == name), key=lambda r: r["gamma"])
        x, y = [r["std_pnl"] for r in sel], [r["mean_pnl"] for r in sel]
        ax.plot(x, y, "-o", color=COLOUR[name], ms=4, label=LABEL[name])
        ax.annotate(f"$\\gamma$={gammas[-1]}", (x[-1], y[-1]), xytext=(0, -12), textcoords="offset points",
                    fontsize=8.5, color=INK_2, ha="center")
    fx = next(r for r in rows if r["strategy"] == "fixed")
    ax.plot(fx["std_pnl"], fx["mean_pnl"], "s", color=COLOUR["fixed"], ms=6, label=LABEL["fixed"])
    ax.annotate(f"$\\gamma$={gammas[0]}", (fx["std_pnl"], fx["mean_pnl"]), xytext=(-6, -14),
                textcoords="offset points", fontsize=8.5, color=INK_2, ha="right")
    ax.set_xlabel("Standard deviation of terminal P&L")
    ax.set_ylabel("Mean terminal P&L")
    legend(ax, loc="lower right")
    title(ax, "The exact optimum cuts risk at almost no cost",
          f"Each point is one $\\gamma$ from {gammas[0]} to {gammas[-1]}; {s['n_paths']:,} paths, dt = {s['dt']}.")
    for name in ("inventory", "exact", "symmetric", "fixed"):
        ax2.plot(gammas, [s["ce_exact"][f"gamma_{g}"][name] for g in gammas], "-o", color=COLOUR[name], ms=4,
                 label=LABEL[name])
    ax2.set_xscale("log")
    ax2.set_ylim(0, 72)
    ax2.set_xticks(gammas, [str(g) for g in gammas])
    ax2.minorticks_off()
    ax2.set_xlabel("Risk aversion $\\gamma$ (log scale)")
    ax2.set_ylabel("Certainty equivalent")
    legend(ax2, loc="lower left", fontsize=8)
    low = s["ce_exact"][f"gamma_{gammas[-1]}"]
    title(ax2, "In utility terms the ranking is never in doubt",
          f"Exact CARA certainty equivalent, inventory limit {s['q_max']}. Symmetric and fixed fall to "
          f"{low['symmetric']:.0f} and {low['fixed']:.0f}.")
    fig.tight_layout()
    save(fig, "risk-aversion-frontier")


def adverse_selection(data: dict, s: dict) -> None:
    rows, sweep = data["rows"], data["sweep"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    for name in ("inventory", "exact", "symmetric", "fixed"):
        sel = [r for r in rows if r["strategy"] == name and r["widen"] == 0]
        ax.plot([r["impact"] for r in sel], [r["mean_pnl"] for r in sel], "-o", color=COLOUR[name], ms=4, label=LABEL[name])
    sel = [r for r in rows if r["strategy"] == "exact" and (r["widen"] > 0 or r["impact"] == 0)]
    ax.plot([r["impact"] for r in sel], [r["mean_pnl"] for r in sel], "--", color=COLOUR["exact"], lw=1.6,
            label="Exact optimum, widened by $\\xi$")
    ax.set_xlabel("Mid-price move per market order, $\\xi$")
    ax.set_ylabel("Mean terminal P&L")
    ax.set_ylim(top=ax.get_ylim()[1] + 3.0)
    legend(ax, loc="upper right", fontsize=8)
    e = s["by_strategy"]["exact"]
    title(ax, f"Informed flow costs every strategy about {e['loss_pct']:.0f}% at $\\xi$ = {s['impact_top']}",
          "Each fill is followed by a move of $\\xi$ against the market maker. Total volatility unchanged.")
    w = [r["widen"] for r in sweep]
    ax2.plot(w, [r["mean_pnl"] for r in sweep], "-o", color=COLOUR["exact"], ms=4)
    ax2.axvline(s["impact_top"], color=MUTED, lw=0.9)
    ax2.annotate("w = $\\xi$", (s["impact_top"], min(r["mean_pnl"] for r in sweep)), xytext=(5, 2),
                 textcoords="offset points", fontsize=9, color=INK_2)
    ax2.set_xlabel("Extra half-spread w added to both quotes")
    ax2.set_ylabel("Mean terminal P&L")
    title(ax2, f"Widening by $\\xi$ is optimal but recovers only {e['recovered_pct_of_loss']:.0f}% of the loss",
          f"Exact optimum at $\\xi$ = {s['impact_top']}; {s['n_paths']:,} paths with common random numbers.")
    fig.tight_layout()
    save(fig, "adverse-selection")


def misspecification(data: dict, s: dict) -> None:
    m, loss = np.array(data["multipliers"]), np.array(data["loss"])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(loss, origin="lower", cmap=SEQ_CMAP, aspect="auto")
    ax.grid(False)
    ticks = [f"{x:.2f}" for x in m]
    ax.set_xticks(range(len(m)), ticks)
    ax.set_yticks(range(len(m)), ticks)
    ax.set_xlabel("Assumed k / true k")
    ax.set_ylabel("Assumed $\\sigma$ / true $\\sigma$")
    for i in range(len(m)):
        for j in range(len(m)):
            ax.text(j, i, f"{loss[i, j]:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if loss[i, j] > 0.5 * loss.max() else INK)
    fig.colorbar(im, ax=ax, pad=0.02).set_label("Certainty equivalent lost")
    title(ax, f"A wrong k costs up to {s['max_loss_k_only']:.1f}; a wrong $\\sigma$ at most {s['max_loss_sigma_only']:.1f}",
          f"Exact strategy built from wrong parameters, evaluated in the true world. Loss out of {s['ce_correct']:.1f}.")
    save(fig, "misspecification")


def main() -> None:
    apply_style()
    summary, exp = read_json("summary.json"), read_json("experiments.json")
    sample_path()
    distributions(summary["replication"])
    quotes_figure(exp["quotes"], summary["quotes"])
    frontier(exp["frontier"], summary["frontier"])
    adverse_selection(exp["adverse_selection"], summary["adverse_selection"])
    misspecification(exp["misspecification"], summary["misspecification"])


if __name__ == "__main__":
    main()
