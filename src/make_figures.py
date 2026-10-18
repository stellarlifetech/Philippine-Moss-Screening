#!/usr/bin/env python3
"""Figures 4-6 of the manuscript, drawn from the analysis outputs.

Usage: python src/make_figures.py results figures
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
FIG = Path(sys.argv[2] if len(sys.argv) > 2 else "figures"); FIG.mkdir(exist_ok=True, parents=True)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
ABBR = {"Sphagnum junghuhnianum": "S. junghuhnianum", "Fissidens nobilis": "F. nobilis",
        "Pyrrhobryum spiniforme": "P. spiniforme", "Entodon plicatus": "E. plicatus",
        "Dendrocyathophorum decolyi": "D. decolyi", "Hyophila involuta": "H. involuta"}


def fig4():
    fig, ax = plt.subplots(figsize=(7.4, 4.2)); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 60)

    def box(x, y, w, h, title, body, fc="#f4f3ee", ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec=MUTED, lw=1, ls=ls))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=6.8, weight="bold", color=INK)
        ax.text(x + w / 2, y + h / 2 - 2.6, body, ha="center", va="center", fontsize=6.6, color=INK, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1, shrinkA=0, shrinkB=0))

    W, H = 21.5, 19
    xs = [1, 26, 51, 76.5]
    box(xs[0], 38, W, H, "1. Image-derived\ninputs", "Raw photomicrographs\n(workflow, Table 2)\nPD, RG, CO, ED\n(Table 3)", fc="#e6f0fb")
    box(xs[0], 13, W, H, "2. Literature-derived\ninputs", "Ordinal scores 0-10\nSC, TO, TR, MC, WH\n(Table 5; rubric Table S1)", fc="#fdebe2")
    box(xs[1], 25.5, W, H, "3. Descriptor\nfusion", "Min-max scaling (Eq. 1)\nD = a(image) + (1 - a)(lit)\na = 0.5 (Eq. 2)\nP, G, U, S; M from lit only\n(Table 6)")
    box(xs[2], 38, W, H, "4a. Heuristic\nprioritization", "H = weighted sum (Eq. 3)\nCPI 0-100 (Eq. 4)\n(Table 7)")
    box(xs[2], 13, W, H, "4b. Areal-density\nproxy", "Exploratory; raw PD only\nPD x density x thickness\n137Cs and 60Co photons\n(Eqs. 5-6; Table 8)")
    box(xs[3], 38, W, H, "6. Robustness", "Fusion ratio, scaling,\nleave-one-out (Table 10)\nMonte Carlo (Table 11)\nSynthetic replication\nsimulation (Table 12)", fc="#e3f4ec")
    box(xs[3], 13, W, H, "5. Two-objective\ntrade space", "Areal-density proxy\nvs hydric buffering M\nPareto front (Table 9)")
    box(26, 1, 72, 7.5, "Output: ranked hypotheses for replicated imaging and laboratory tests.",
        "", fc="#ffffff", ls="--")
    ax.text(62, 2.6, "No material was fabricated, and no radiation was measured.", ha="center", fontsize=6.6, color=MUTED)
    arrow(xs[0] + W + 0.6, 47.5, xs[1] - 0.6, 40)
    arrow(xs[0] + W + 0.6, 22.5, xs[1] - 0.6, 30)
    arrow(xs[1] + W + 0.6, 40, xs[2] - 0.6, 47.5)
    arrow(xs[1] + W + 0.6, 30, xs[2] - 0.6, 22.5)
    arrow(xs[2] + W + 0.6, 47.5, xs[3] - 0.6, 47.5)
    arrow(xs[2] + W + 0.6, 22.5, xs[3] - 0.6, 22.5)
    fig.savefig(FIG / "figure4_framework.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def fig5():
    t9 = pd.read_csv(OUT / "table09_pareto.csv", index_col=0)
    g = pd.read_csv(OUT / "table11_mc_global.csv", index_col=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.1, 1]})
    for i, (sp, r) in enumerate(t9.iterrows()):
        opt = r["status"] == "Pareto-optimal"
        a.scatter(r["areal_density_g_cm2"], r["theta_WH"], s=46, color=C[i], edgecolor="white",
                  lw=1.5, zorder=3, marker="o" if opt else "s")
        off = (-4, -12) if sp == "Pyrrhobryum spiniforme" else (5, 4)
        a.annotate(ABBR[sp], (r["areal_density_g_cm2"], r["theta_WH"]), xytext=off,
                   textcoords="offset points", fontsize=7.2, style="italic", color=INK)
    front = t9[t9.status == "Pareto-optimal"].sort_values("areal_density_g_cm2")
    a.plot(front["areal_density_g_cm2"], front["theta_WH"], color=MUTED, lw=1, ls="--", zorder=2)
    a.set_xlabel("Areal-density proxy (g cm$^{-2}$, assumed t = 0.30 cm)")
    a.set_ylabel("Hydric buffering score M (0-1)")
    a.set_xlim(0.072, 0.122); a.set_ylim(-0.08, 1.12); a.grid(color=GRID, lw=0.6)
    a.set_title("A. Two-objective trade space", loc="left", fontsize=9, weight="bold")
    a.text(0.073, -0.06, "circle = Pareto-optimal; square = dominated", fontsize=6.5, color=MUTED)
    order = g.sort_values("mean_rank").index
    for j, sp in enumerate(order):
        i = list(t9.index).index(sp)
        r = g.loc[sp]
        b.plot([r["rank_2.5"], r["rank_97.5"]], [j, j], color=C[i], lw=2, solid_capstyle="round")
        b.scatter(r["mean_rank"], j, s=40, color=C[i], edgecolor="white", lw=1.5, zorder=3)
        b.text(6.35, j, (f"{r['P_top3_%']:.0f}%" if r['P_top3_%'] < 99.5 else ">99%"), va="center", fontsize=7, color=INK)
    b.text(6.35, -0.75, "P(top 3)", fontsize=6.5, color=MUTED)
    b.set_yticks(range(len(order))); b.set_yticklabels([ABBR[s] for s in order], style="italic")
    b.set_ylim(5.6, -1.1); b.set_xlim(0.7, 6.3); b.set_xticks(range(1, 7))
    b.set_xlabel("Rank under global perturbation\n(mean and 95% simulation interval)")
    b.grid(axis="x", color=GRID, lw=0.6)
    b.set_title("B. Rank stability (10,000 iterations)", loc="left", fontsize=9, weight="bold")
    fig.tight_layout(); fig.savefig(FIG / "figure5_tradespace_stability.png", dpi=300); plt.close(fig)


def fig6():
    t = pd.read_csv(OUT / "table12_synthetic_replication_SIMULATED.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, col, ttl in zip(axs, ["P_same_top3_set_%", "P_exact_order_%"],
                            ["A. Same top-three set", "B. Same full rank order"]):
        for k, cv in enumerate(sorted(t.CV_between.unique())):
            s = t[t.CV_between == cv]
            ax.plot(s.specimens_per_species, s[col], color=C[k], lw=2, marker="o", ms=5,
                    mec="white", mew=1.2, label=f"between-specimen CV {cv:.0%}")
            ax.text(10.3, s[col].iloc[-1], f"{cv:.0%}", va="center", fontsize=7, color=INK)
        ax.set_xticks([1, 3, 5, 10]); ax.set_xlim(0.5, 11.5); ax.set_ylim(0, 100)
        ax.set_xlabel("Specimens per species (4 fields each)"); ax.grid(color=GRID, lw=0.6)
        ax.set_title(ttl, loc="left", fontsize=9, weight="bold")
    axs[0].set_ylabel("Simulated recovery probability (%)")
    axs[0].legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.text(0.5, -0.02, "Synthetic design simulation: simulated values, not measurements.",
             ha="center", fontsize=6.8, color=MUTED)
    fig.tight_layout(); fig.savefig(FIG / "figure6_synthetic_replication.png", dpi=300, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig4(); fig5(); fig6(); print("figures written to", FIG.resolve())
