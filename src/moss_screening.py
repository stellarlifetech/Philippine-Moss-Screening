#!/usr/bin/env python3
"""
Computational screening of Philippine moss morphology for biomimetic
shielding-oriented design: revised, species-keyed analysis.

Every table in the revised manuscript (Tables 5-13 and S1-S2) is regenerated
from the two species-keyed input tables below. No value is copied by row
position: every computation indexes species by name, and automated
consistency tests (run_consistency_tests) stop the run if any output row
cannot be traced back to the input row of the same species.

Usage:
    python src/moss_screening.py                       # data/ -> results/
    python src/moss_screening.py --data data --out results --iterations 10000
Requires: Python >= 3.11, numpy, pandas, scipy (see requirements.txt).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED = 20261018          # fixed seed for every stochastic analysis
N_ITER = 10_000          # Monte Carlo iterations
N_REPS = 2_000           # replicates per scenario, synthetic replication simulation

SPECIES = [
    "Sphagnum junghuhnianum",
    "Fissidens nobilis",
    "Pyrrhobryum spiniforme",
    "Entodon plicatus",
    "Dendrocyathophorum decolyi",
    "Hyophila involuta",
]

# ---------------------------------------------------------------------------
# INPUTS (species-keyed CSV files in data/)
#   image_descriptors.csv : Table 3, one photomicrograph per species (n = 1)
#   literature_scores.csv : Table 5, ordinal 0-10 scores (rubric Table S1)
# ---------------------------------------------------------------------------
IMAGE_COLS = ["PD", "OSF", "RG", "CO", "LI", "ED"]
LIT_COLS = ["SC", "TO", "TR", "MC", "WH"]
IMAGE = pd.DataFrame()
LIT = pd.DataFrame()


def load_inputs(data_dir: Path) -> None:
    """Read both input tables, indexed by species name (never by row order)."""
    global IMAGE, LIT
    img = pd.read_csv(data_dir / "image_descriptors.csv").set_index("species")
    lit = pd.read_csv(data_dir / "literature_scores.csv").set_index("species")
    missing = [c for c in IMAGE_COLS if c not in img] + [c for c in LIT_COLS if c not in lit]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    if img.index.duplicated().any() or lit.index.duplicated().any():
        raise ValueError("duplicate species rows in input")
    IMAGE, LIT = img[IMAGE_COLS].astype(float), lit[LIT_COLS].astype(float)


# Pairing of image and literature components for each fused descriptor.
# Layering Index (LI) is excluded: its observed range (0.5050-0.5062) is
# 0.24% of its mean and min-max scaling would inflate that noise to 0-1.
FUSION = {"P": ("PD", "SC"), "G": ("RG", "TO"), "U": ("CO", "TR"), "S": ("ED", "MC")}

# Heuristic weights: original Eq. 3 weights with radionuclide affinity
# removed and the remaining five rescaled to sum to 1 (0.22,0.18,0.16,0.14,
# 0.20)/0.90, rounded to two decimals.
WEIGHTS = {"P": 0.24, "G": 0.20, "U": 0.18, "S": 0.16, "M": 0.22}
STRUCT = ["P", "G", "U", "S"]

# Exploratory areal-density proxy parameters (assumed, not measured)
T_CM = 0.30              # nominal specimen thickness (cm)
RHO_T = 1.00             # hydrated tissue density, water-equivalent (g/cm3)
# Water mass-attenuation coefficients, NIST XCOM / SRD 126 (cm2/g)
MU_RHO = {"0.662 MeV (137Cs)": None, "1.25 MeV (60Co)": 0.06323}
_E, _MU = np.array([0.6, 0.8]), np.array([0.08956, 0.07865])  # water, NIST
MU_RHO["0.662 MeV (137Cs)"] = float(np.exp(np.interp(np.log(0.662), np.log(_E), np.log(_MU))))


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------
def minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0.0


def robust(s: pd.Series) -> pd.Series:
    """Median/IQR scaling followed by a logistic squashing to (0,1)."""
    iqr = s.quantile(0.75) - s.quantile(0.25)
    z = (s - s.median()) / (iqr if iqr > 0 else 1.0)
    return 1.0 / (1.0 + np.exp(-z))


def fuse(image: pd.DataFrame, lit: pd.DataFrame, alpha: float = 0.5,
         scaler=minmax) -> pd.DataFrame:
    """Return fused descriptors P,G,U,S,M keyed by species (Eqs. 1-2)."""
    out = pd.DataFrame(index=image.index)
    for d, (ic, lc) in FUSION.items():
        out[d] = alpha * scaler(image[ic]) + (1 - alpha) * scaler(lit.loc[image.index, lc])
    out["M"] = scaler(lit.loc[image.index, "WH"])   # literature only
    return out


def heuristic(F: pd.DataFrame, w: dict = WEIGHTS) -> pd.Series:
    return sum(w[k] * F[k] for k in w)


def structural(F: pd.DataFrame, w: dict = WEIGHTS) -> pd.Series:
    tot = sum(w[k] for k in STRUCT)
    return sum(w[k] * F[k] for k in STRUCT) / tot


def cpi(h: pd.Series) -> pd.Series:
    return 100 * minmax(h)


def ranks(s: pd.Series) -> pd.Series:
    return s.rank(ascending=False, method="min").astype(int)


def areal(image: pd.DataFrame, t=T_CM, rho=RHO_T) -> pd.Series:
    """Areal-density proxy (g/cm2): Delesse area fraction x density x thickness."""
    return image["PD"] * rho * t


def pareto(a: pd.Series, b: pd.Series) -> dict:
    status = {}
    for i in a.index:
        dom = [j for j in a.index if j != i and a[j] >= a[i] and b[j] >= b[i]
               and (a[j] > a[i] or b[j] > b[i])]
        status[i] = dom
    return status


# ---------------------------------------------------------------------------
# Consistency tests (guard against the row-mapping error in the earlier draft)
# ---------------------------------------------------------------------------
def run_consistency_tests(F: pd.DataFrame) -> None:
    assert set(IMAGE.index) == set(LIT.index) == set(SPECIES), "species sets differ"
    assert abs(sum(WEIGHTS.values()) - 1) < 1e-9
    assert np.allclose(IMAGE["OSF"], 1 - IMAGE["PD"], atol=1e-4)
    assert ((F >= -1e-12) & (F <= 1 + 1e-12)).all().all()
    # Each fused value must be recomputable from the SAME species' inputs
    for sp in SPECIES:
        for d, (ic, lc) in FUSION.items():
            img = (IMAGE.at[sp, ic] - IMAGE[ic].min()) / (IMAGE[ic].max() - IMAGE[ic].min())
            lit = (LIT.at[sp, lc] - LIT[lc].min()) / (LIT[lc].max() - LIT[lc].min())
            assert abs(F.at[sp, d] - (0.5 * img + 0.5 * lit)) < 1e-12, (sp, d)
    # Species with the extreme raw input must hold the extreme normalized component
    assert F["S"].idxmax() == "Dendrocyathophorum decolyi"  # max ED and max MC
    assert IMAGE["PD"].idxmax() == "Entodon plicatus"


# ---------------------------------------------------------------------------
# Robustness analyses
# ---------------------------------------------------------------------------
def summarize_ranks(R: np.ndarray, names) -> pd.DataFrame:
    df = pd.DataFrame({
        "mean_rank": R.mean(0),
        "rank_2.5": np.percentile(R, 2.5, axis=0),
        "rank_97.5": np.percentile(R, 97.5, axis=0),
        "P_rank1_%": (R == 1).mean(0) * 100,
        "P_top3_%": (R <= 3).mean(0) * 100,
    }, index=names)
    return df


def mc(kind: str, rng: np.random.Generator, n=None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    n = N_ITER if n is None else n
    names = SPECIES
    R = np.zeros((n, len(names)), int)
    H = np.zeros((n, len(names)))
    keys = list(WEIGHTS)
    w0 = np.array([WEIGHTS[k] for k in keys])
    for it in range(n):
        img, lit, w = IMAGE.copy(), LIT.copy(), dict(WEIGHTS)
        alpha = 0.5
        if kind in ("image", "global"):
            for c in ["PD", "RG", "CO", "ED"]:
                img[c] = img[c] * (1 + rng.normal(0, 0.10, len(img)))
        if kind in ("ordinal", "global"):
            for c in ["SC", "TO", "TR", "MC", "WH"]:
                lit[c] = np.clip(lit[c] + rng.integers(-1, 2, len(lit)), 0, 10)
        if kind in ("weights", "global"):
            ww = rng.dirichlet(50 * w0)
            w = dict(zip(keys, ww))
        if kind == "weights_flat":
            w = dict(zip(keys, rng.dirichlet(np.ones(len(keys)))))
        if kind in ("fusion", "global"):
            alpha = rng.uniform(0.3, 0.7)
        F = fuse(img, lit, alpha)
        h = heuristic(F, w).loc[names]
        H[it] = h.values
        R[it] = ranks(h).values
    return summarize_ranks(R, names), R, H


def replication_simulation(rng, reps=None, k_fields=4) -> pd.DataFrame:
    """SYNTHETIC design simulation (not data): how many specimens per species
    are needed for the image-derived inputs to resolve the ranking, given
    assumed between-specimen and within-specimen coefficients of variation?"""
    reps = N_REPS if reps is None else reps
    base_rank = ranks(heuristic(fuse(IMAGE, LIT)))
    rows = []
    for cvb in (0.10, 0.20, 0.30):
        for n_spec in (1, 3, 5, 10):
            top, rho, exact, top3 = 0, [], 0, 0
            for _ in range(reps):
                img = IMAGE.copy()
                for c in ["PD", "RG", "CO", "ED"]:
                    mu = IMAGE[c].values[:, None, None]
                    spec = mu * (1 + rng.normal(0, cvb, (len(IMAGE), n_spec, 1)))
                    field = spec * (1 + rng.normal(0, 0.10, (len(IMAGE), n_spec, k_fields)))
                    img[c] = field.mean(2).mean(1)      # specimen = biological unit
                r = ranks(heuristic(fuse(img, LIT)))
                top += r.idxmin() == base_rank.idxmin()
                exact += (r == base_rank).all()
                top3 += set(r[r <= 3].index) == set(base_rank[base_rank <= 3].index)
                rho.append(spearmanr(r.loc[SPECIES], base_rank.loc[SPECIES])[0])
            rows.append({"CV_between": cvb, "specimens_per_species": n_spec,
                         "fields_per_specimen": k_fields,
                         "P_same_top_%": 100 * top / reps,
                         "P_same_top3_set_%": 100 * top3 / reps,
                         "P_exact_order_%": 100 * exact / reps,
                         "median_spearman": float(np.median(rho))})
    return pd.DataFrame(rows)


def extra_sensitivity(out: Path) -> dict:
    """Additional baselines requested in review (Table 10c-e, Table 11b).

    * leave-one-descriptor-out: each fused descriptor removed in turn, the
      remaining weights rescaled to sum to one;
    * unweighted rank-sum (Borda) baseline: species ranked on each fused
      descriptor and the ranks summed, with no weights and no scaling;
    * single-source baselines: image-only (alpha = 1, P/G/U/S only, equal
      weights) and literature-only (alpha = 0, all five, equal weights);
    * wider ordinal uncertainty: every literature score shifted by an integer
      in [-2, 2] (10,000 iterations, separate seed so earlier outputs are unchanged).
    """
    F = fuse(IMAGE, LIT)
    lodo = {"baseline": ranks(heuristic(F))}
    for k in WEIGHTS:
        w = {j: v for j, v in WEIGHTS.items() if j != k}
        tot = sum(w.values())
        lodo[f"without {k}"] = ranks(heuristic(F, {j: v / tot for j, v in w.items()}))
    t_lodo = pd.DataFrame(lodo).loc[SPECIES]
    t_lodo.to_csv(out / "table10c_leave_one_descriptor_out.csv")

    desc_ranks = F[list(WEIGHTS)].rank(ascending=False, method="average")
    borda = desc_ranks.sum(axis=1)
    eq = {k: 0.2 for k in WEIGHTS}
    t_base = pd.DataFrame({
        "weighted_baseline": ranks(heuristic(F)),
        "equal_weights": ranks(heuristic(F, eq)),
        "rank_sum_score": borda,
        "rank_sum_unweighted": borda.rank(method="min").astype(int),
        # image-only: alpha = 1 and the literature-only descriptor M excluded
        "image_only_equal_weights": ranks(heuristic(fuse(IMAGE, LIT, 1.0),
                                                    {"P": .25, "G": .25, "U": .25, "S": .25, "M": 0.0})),
        "literature_only_equal_weights": ranks(heuristic(fuse(IMAGE, LIT, 0.0), eq)),
    }).loc[SPECIES]
    t_base.to_csv(out / "table10d_unweighted_baselines.csv")

    rng2 = np.random.default_rng(SEED + 1)
    R = np.zeros((N_ITER, len(SPECIES)), int)
    for it in range(N_ITER):
        lit = LIT.copy()
        for c in ["SC", "TO", "TR", "MC", "WH"]:
            lit[c] = np.clip(lit[c] + rng2.integers(-2, 3, len(lit)), 0, 10)
        R[it] = ranks(heuristic(fuse(IMAGE, lit))).loc[SPECIES].values
    t_o2 = summarize_ranks(R, SPECIES).round(2)
    t_o2.to_csv(out / "table11b_mc_ordinal_pm2.csv")
    return {"ordinal_pm2_seed": SEED + 1}


# ---------------------------------------------------------------------------
def main(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {}

    F = fuse(IMAGE, LIT)
    run_consistency_tests(F)
    H = heuristic(F)
    SS = structural(F)
    t5 = F.loc[SPECIES].round(4)
    t6 = pd.DataFrame({"structural_score": SS, "hydric_M": F["M"], "H_raw": H,
                       "CPI": cpi(H), "rank": ranks(H)}).sort_values("rank").round(4)
    eq = heuristic(F, {k: 0.2 for k in WEIGHTS})
    t6["H_equal_weight"] = eq.round(4)
    t6["rank_equal_weight"] = ranks(eq)
    t6["rank_structural_only"] = ranks(SS)
    t6["rank_hydric_only"] = ranks(F["M"])
    t5.to_csv(out / "table06_fused_descriptors.csv")
    t6.to_csv(out / "table07_heuristic.csv")

    # Exploratory areal-density proxy (Table 7)
    A = areal(IMAGE)
    t7 = pd.DataFrame({"area_fraction_PD": IMAGE["PD"], "areal_density_g_cm2": A})
    for lab, mu in MU_RHO.items():
        t7[f"T_{lab}"] = np.exp(-mu * A)
        t7[f"B_{lab}"] = 1 - np.exp(-mu * A)
    t7["rank_areal"] = ranks(A)
    t7["theta_WH"] = F["M"]
    t7 = t7.loc[SPECIES].round(4)
    t7.to_csv(out / "table08_areal_density.csv")

    # Pareto (Table 8): areal density (image only) vs hydric buffering (literature only)
    P = pareto(A, F["M"])
    t8 = pd.DataFrame({"areal_density_g_cm2": A.round(4), "theta_WH": F["M"].round(4),
                       "status": ["Pareto-optimal" if not P[s] else "Dominated" for s in A.index],
                       "dominated_by": ["; ".join(P[s]) or "-" for s in A.index]}).loc[SPECIES]
    t8.to_csv(out / "table09_pareto.csv")

    # Fusion-ratio sweep (Table 9)
    sweep = {}
    for a in np.round(np.arange(0, 1.01, 0.1), 1):
        sweep[a] = ranks(heuristic(fuse(IMAGE, LIT, a)))
    t9 = pd.DataFrame(sweep).loc[SPECIES]
    t9.to_csv(out / "table10a_fusion_sweep.csv")

    # Scaling alternative and leave-one-species-out (Table 10)
    rob = ranks(heuristic(fuse(IMAGE, LIT, scaler=robust)))
    loso = {}
    for drop in SPECIES:
        keep = [s for s in SPECIES if s != drop]
        loso[f"without {drop}"] = ranks(heuristic(fuse(IMAGE.loc[keep], LIT.loc[keep])))
    t10 = pd.DataFrame({"baseline": ranks(H), "robust_scaling": rob, **loso}).loc[SPECIES]
    t10.to_csv(out / "table10b_scaling_loso.csv")

    # Local sensitivity (elasticity), Table S2
    el = {}
    for k in WEIGHTS:
        e = []
        for d in (+0.10, -0.10):
            F2 = F.copy()
            F2[k] = np.clip(F2[k] * (1 + d), 0, 1)
            h2 = heuristic(F2)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = ((h2 - H) / H) / (((F2[k] - F[k]) / F[k]).replace(0, np.nan))
            e.append(ratio.mean())
        el[k] = {"weight": WEIGHTS[k], "mean_elasticity": float(np.nanmean(e))}
    pd.DataFrame(el).T.round(4).to_csv(out / "tableS2_elasticity.csv")

    # Monte Carlo (Tables 11-12)
    mc_tabs = {}
    for kind in ["image", "ordinal", "weights", "weights_flat", "fusion", "global"]:
        tab, R, Hs = mc(kind, rng)
        tab.round(2).to_csv(out / f"table11_mc_{kind}.csv")
        mc_tabs[kind] = tab
        if kind == "global":
            # convergence check: P(rank1) for Sphagnum at increasing n
            conv = {n: float((R[:n, 0] == 1).mean() * 100) for n in (1000, 2500, 5000, 10000) if n <= N_ITER}
            res["convergence_P_rank1_Sphagnum"] = conv
            hs = pd.DataFrame({"H_mean": Hs.mean(0), "H_2.5": np.percentile(Hs, 2.5, 0),
                               "H_97.5": np.percentile(Hs, 97.5, 0)}, index=SPECIES)
            hs.round(4).to_csv(out / "table11_global_scores.csv")
    t11 = pd.concat({k: v["P_rank1_%"] for k, v in mc_tabs.items()}, axis=1).round(1)
    t11.to_csv(out / "table11_summary_P_rank1.csv")

    # Synthetic replication design simulation (Table 12)
    t12 = replication_simulation(rng)
    t12.round({"P_same_top_%": 1, "P_same_top3_set_%": 1, "P_exact_order_%": 1,
               "median_spearman": 3}).to_csv(out / "table12_synthetic_replication_SIMULATED.csv", index=False)

    res.update(extra_sensitivity(out))
    res.update({
        "seed": SEED, "iterations": N_ITER, "replicates_per_scenario": N_REPS, "mu_rho": MU_RHO,
        "spearman_heuristic_vs_areal": float(spearmanr(H.loc[SPECIES], A.loc[SPECIES])[0]),
        "spearman_structural_vs_hydric": float(spearmanr(SS.loc[SPECIES], F["M"].loc[SPECIES])[0]),
    })
    (out / "run_metadata.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--data", type=Path, default=root / "data")
    ap.add_argument("--out", type=Path, default=root / "results")
    ap.add_argument("--iterations", type=int, default=N_ITER,
                    help="Monte Carlo iterations (manuscript: 10000)")
    ap.add_argument("--reps", type=int, default=N_REPS,
                    help="replicates per synthetic scenario (manuscript: 2000)")
    a = ap.parse_args()
    N_ITER, N_REPS = a.iterations, a.reps
    load_inputs(a.data)
    print(json.dumps(main(a.out), indent=2))
    print("All consistency tests passed; outputs written to", a.out.resolve())
