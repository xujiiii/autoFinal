from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METHODS = [
    ("lgbm_gain",         "LightGBM gain"),
    ("lgbm_shap_meanabs", "SHAP |abs|"),
    ("lgbm_permutation",  "Permutation"),
    ("rf_impurity",       "RF impurity"),
]
TREE_METHODS = ["lgbm_gain", "lgbm_shap_meanabs",
                "lgbm_permutation", "rf_impurity"]
N_LIST = [10, 20, 50, 100, 200]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# RESIDUE-LEVEL aggregation
def per_residue_score(df: pd.DataFrame, col: str, top_k: int | None = None
                      ) -> pd.Series:
    """Aggregate pair-level scores to per-residue scores.

    Sum of 1/rank (over the full pair set or the top_k pairs touching
    that residue, if top_k given)."""
    d = df.copy()
    d["rank"] = d[col].abs().rank(method="average", ascending=False)
    if top_k is not None:
        d = d.nsmallest(top_k, "rank")
    d["w"] = 1.0 / d["rank"]
    a = d.groupby("resi_i")["w"].sum()
    b = d.groupby("resi_j")["w"].sum()
    return a.add(b, fill_value=0)


def residue_level_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Spearman ρ + Jaccard@K_resi between every pair of methods, at the
    *residue* level instead of the *pair* level."""
    cols = [m for m, _ in METHODS]
    scores = {c: per_residue_score(df, c) for c in cols}
    all_resi = sorted(set().union(*[s.index for s in scores.values()]))
    score_mat = pd.DataFrame(
        {c: scores[c].reindex(all_resi).fillna(0) for c in cols},
        index=all_resi)

    rows = []
    K_resi = [5, 10, 15, 20, 30]
    for a, b in combinations(cols, 2):
        rho, _ = spearmanr(score_mat[a], score_mat[b])
        row = {"target": target, "method_a": a, "method_b": b,
               "spearman_residue": float(rho)}
        for k in K_resi:
            ta = set(score_mat[a].nlargest(k).index)
            tb = set(score_mat[b].nlargest(k).index)
            row[f"jaccard_top{k}_resi"] = (
                len(ta & tb) / max(len(ta | tb), 1))
        rows.append(row)
    return pd.DataFrame(rows)


def within_method_stability(df: pd.DataFrame, target: str,
                            n_boot: int = 20, seed: int = 25
                            ) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_feat = len(df)
    rows = []
    for m, _ in METHODS:
        s = df[m].abs().values
        full_top = {n: set(np.argsort(-s)[:n]) for n in N_LIST}
        jacs = {n: [] for n in N_LIST}
        for _ in range(n_boot):
            idx = rng.integers(0, n_feat, size=n_feat)
            # Bootstrap the FEATURE LIST.  Rank within the bootstrap.
            ss = s[idx]
            # Map a feature's bootstrap-rank back by averaging over
            # duplicate appearances.
            r = pd.Series(ss).rank(method="average", ascending=False)
            # Aggregate per original feature index
            avg_rank = pd.Series(r.values, index=idx).groupby(level=0).mean()
            for n in N_LIST:
                top_boot = set(avg_rank.nsmallest(n).index)
                top_full = full_top[n]
                jacs[n].append(
                    len(top_boot & top_full)
                    / max(len(top_boot | top_full), 1))
        row = {"target": target, "method": m}
        for n in N_LIST:
            row[f"jaccard_top{n}_self_boot_mean"] = float(np.mean(jacs[n]))
            row[f"jaccard_top{n}_self_boot_std"] = float(np.std(jacs[n]))
        rows.append(row)
    return pd.DataFrame(rows)


def is_neighbor_match(pair_a: tuple, pair_b: tuple) -> bool:
    ai, aj = pair_a
    bi, bj = pair_b

    if ai == bi and abs(aj - bj) <= 2:
        return True
   
    if aj == bj and abs(ai - bi) <= 2:
        return True

    if ai == bj and abs(aj - bi) <= 2:
        return True
    if aj == bi and abs(ai - bj) <= 2:
        return True

    return False

#The table used to plot cluster level vs pair level jaccard
def cluster_level_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    cols = [m for m, _ in METHODS]

    pairs = list(zip(df["resi_i"], df["resi_j"]))

    method_sorted_pairs = {}
    for c in cols:
        sorted_indices = df[c].sort_values(ascending=False).index
        method_sorted_pairs[c] = [pairs[idx] for idx in sorted_indices]

    rows = []
    K_clust = [5, 10, 20, 50]

    for a, b in combinations(cols, 2):
        pairs_a_full = method_sorted_pairs[a]
        pairs_b_full = method_sorted_pairs[b]

        rho, _ = spearmanr(df[a], df[b])

        row = {
            "target": target,
            "method_a": a,
            "method_b": b,
            "n_clusters": int(len(df)),
            "spearman_cluster": float(rho),
        }

        for k in K_clust:
            ta_list = pairs_a_full[:k]
            tb_list = pairs_b_full[:k]
            intersection_count = 0
            for aa in ta_list:
                for bb in tb_list:
                    if is_neighbor_match(aa, bb):
                        intersection_count += 1
                        break  

            union_set = set(ta_list) | set(tb_list)
            union_count = len(union_set)

            row[f"jaccard_top{k}_cluster"] = intersection_count / max(
                union_count, 1
            )

        rows.append(row)

    return pd.DataFrame(rows)


def plot_residue_vs_pair(res_df_all: pd.DataFrame, pair_df_all: pd.DataFrame,
                         out: Path):
    # average over targets
    pair = (pair_df_all.groupby(["method_a", "method_b"])
            ["jaccard_top20"].mean().reset_index())
    res = (res_df_all.groupby(["method_a", "method_b"])
           ["jaccard_top20_resi"].mean().reset_index())
    df = pair.merge(res, on=["method_a", "method_b"])
    df["label"] = df["method_a"].str.replace("lgbm_", "").str.replace(
        "_meanabs", "") + " ↔ " + df["method_b"].str.replace(
        "lgbm_", "").str.replace("_meanabs", "")

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(df))
    ax.barh(x - 0.2, df["jaccard_top20"], height=0.4,
             color="#C62828", label="Pair-level top-20")
    ax.barh(x + 0.2, df["jaccard_top20_resi"], height=0.4,
             color="#1565C0", label="Residue-level top-20")
    ax.set_yticks(x); ax.set_yticklabels(df["label"], fontsize=14)
    ax.set_xlabel("Jaccard overlap (z0 + z1 averaged)", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_title("Pair-level vs residue-level Jaccard",
                  fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fi_residue_vs_pair_jaccard.{ext}",
                    dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_within_vs_between(within_df: pd.DataFrame,
                            pair_agreement: pd.DataFrame,
                            out: Path):
    """The headline diagnostic: within-method bootstrap Jaccard@N
    (every method's intrinsic noise floor) overlaid on the
    between-method Jaccard@N from the original analysis."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10")
    label_map = dict(METHODS)
    for k, (m, _) in enumerate(METHODS):
        # average within-method bootstrap across z0+z1
        sub = within_df[within_df["method"] == m]
        ys = [sub[f"jaccard_top{n}_self_boot_mean"].mean() for n in N_LIST]
        es = [sub[f"jaccard_top{n}_self_boot_std"].mean() for n in N_LIST]
        ax.errorbar(N_LIST, ys, yerr=es, marker="o", capsize=3,
                     lw=2, color=cmap(k % 10), label=f"{label_map[m]} (self)")

    # Between-method: SHAP ↔ everything else, averaged over z0, z1
    pa = (pair_agreement.set_index(["method_a", "method_b"]))
    pa = pa.sort_index()
    for m in [mm for mm, _ in METHODS if mm != "lgbm_shap_meanabs"]:
        key = ("lgbm_shap_meanabs", m) if (
            ("lgbm_shap_meanabs", m) in pa.index) else (m, "lgbm_shap_meanabs")
        if key not in pa.index:
            continue
        ys = []
        for n in N_LIST:
            try:
                ys.append(pa.loc[key, f"jaccard_top{n}"].mean())
            except KeyError:
                ys.append(np.nan)
        ax.plot(N_LIST, ys, marker="s", lw=1.3, ls="--",
                 color="0.4", alpha=0.4,
                 label=f"SHAP ↔ {label_map[m]} (between)")

    ax.set_xscale("log")
    ax.set_xlabel("Top-N")
    ax.set_ylabel("Jaccard@N")
    ax.set_title("Within-method bootstrap stability (solid) vs\n"
                  "between-method agreement against SHAP (dashed)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=2, loc="upper left",
               bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fi_within_vs_between.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


#Plot the cluster level vs pair level graph
def plot_cluster_vs_pair(
    clust_df_all: pd.DataFrame, pair_df_all: pd.DataFrame, out: Path
):
    pair = (
        pair_df_all.groupby(["method_a", "method_b"])["jaccard_top20"]
        .mean()
        .reset_index()
    )
    clust = (
        clust_df_all.groupby(["method_a", "method_b"])["jaccard_top20_cluster"]
        .mean()
        .reset_index()
    )
    df = pair.merge(clust, on=["method_a", "method_b"])
    df["label"] = (
        df["method_a"].str.replace("lgbm_", "").str.replace("_meanabs", "")
        + " ↔ "
        + df["method_b"].str.replace("lgbm_", "").str.replace("_meanabs", "")
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(df))
    ax.barh(
        x - 0.2,
        df["jaccard_top20"],
        height=0.4,
        color="#C62828",
        label="Raw pair-level top-20",
    )
    ax.barh(
        x + 0.2,
        df["jaccard_top20_cluster"],
        height=0.4,
        color="#EE00FF",
        label="Cluster-level top-20",
    )
    ax.set_yticks(x)
    ax.set_yticklabels(df["label"], fontsize=14)
    ax.set_xlabel("Jaccard overlap (z0 + z1 averaged)",fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_title("Pair-level vs cluster-level Jaccard agreement", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            out / f"fi_cluster_vs_pair_jaccard.{ext}",
            dpi=400,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended-fi-csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out / "figures"; fig_dir.mkdir(exist_ok=True)
    full = pd.read_csv(args.extended_fi_csv)
    res_rows, stab_rows, clust_rows = [], [], []
    for t in ("z0", "z1"):
        df = full[full["target"] == t].reset_index(drop=True).copy()
        for m, _ in METHODS:
            df[m] = df[m].abs()
        res_rows.append(residue_level_table(df, t))
        stab_rows.append(within_method_stability(df, t,
                                                  n_boot=args.n_boot))
        clust_rows.append(cluster_level_table(df, t))
    res_df = pd.concat(res_rows, ignore_index=True)
    stab_df = pd.concat(stab_rows, ignore_index=True)
    clust_df = pd.concat(clust_rows, ignore_index=True)
    res_df.to_csv(args.out / "fi_residue_level_agreement.csv", index=False)
    stab_df.to_csv(args.out / "fi_within_method_stability.csv", index=False)
    clust_df.to_csv(args.out / "fi_cluster_level_agreement.csv", index=False)

    from glob import glob
    import os

    pattern = os.path.join(str(args.out), "fi_method_agreement_z?.csv")

    pa_files = sorted(glob(pattern))
    if pa_files:
        pair_agreement = pd.concat([pd.read_csv(p) for p in pa_files],
                                    ignore_index=True)
    else:
        print("WARNING: no pair-agreement CSVs found, skipping "
              "within-vs-between plot")
        pair_agreement = None

    plot_residue_vs_pair(res_df, pair_agreement, fig_dir) if pair_agreement is not None else None
    plot_cluster_vs_pair(clust_df, pair_agreement, fig_dir) if pair_agreement is not None else None
    if pair_agreement is not None:
        plot_within_vs_between(stab_df, pair_agreement, fig_dir)

    print("\n=== HEADLINE ===")
    print("\nResidue-level Spearman (averaged over z0, z1):")
    pivot = (res_df.groupby(["method_a", "method_b"])
              ["spearman_residue"].mean().reset_index()
              .sort_values("spearman_residue", ascending=False))
    print(pivot.to_string(index=False))

    print("\nResidue-level Jaccard@10 (averaged over z0, z1):")
    pivot = (res_df.groupby(["method_a", "method_b"])
              ["jaccard_top10_resi"].mean().reset_index()
              .sort_values("jaccard_top10_resi", ascending=False))
    print(pivot.to_string(index=False))

    print("\nWithin-method bootstrap Jaccard@20 (lower = noisier):")
    pivot = (stab_df.groupby("method")
              ["jaccard_top20_self_boot_mean"].mean().sort_values())
    print(pivot.to_string())

    print("\nCluster-level Jaccard@20 (averaged over z0, z1):")
    pivot = (clust_df.groupby(["method_a", "method_b"])
              ["jaccard_top20_cluster"].mean().reset_index()
              .sort_values("jaccard_top20_cluster", ascending=False))
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
