from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

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


def agreement_table(df: pd.DataFrame) -> dict:
    """Build pairwise Spearman, Pearson and Jaccard@N for one target."""
    cols = [m for m, _ in METHODS]
    score = {c: df[c].values for c in cols}
    rank = {c: pd.Series(s).rank(method="average", ascending=False).values
            for c, s in score.items()}
    n_feat = len(df)

    sp = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pe = pd.DataFrame(index=cols, columns=cols, dtype=float)
    jac = {n: pd.DataFrame(index=cols, columns=cols, dtype=float)
           for n in N_LIST}
    for a, b in combinations(cols, 2):
        rho, _ = spearmanr(score[a], score[b])
        r, _ = pearsonr(score[a], score[b])
        sp.loc[a, b] = sp.loc[b, a] = rho
        pe.loc[a, b] = pe.loc[b, a] = r
        ra = rank[a]; rb = rank[b]
        for n in N_LIST:
            top_a = set(np.where(ra <= n)[0])
            top_b = set(np.where(rb <= n)[0])
            jac[n].loc[a, b] = jac[n].loc[b, a] = (
                len(top_a & top_b) / max(len(top_a | top_b), 1))
    for c in cols:
        sp.loc[c, c] = 1.0
        pe.loc[c, c] = 1.0
        for n in N_LIST:
            jac[n].loc[c, c] = 1.0
    return {"spearman": sp, "pearson": pe, "jaccard": jac,
            "n_features": n_feat}


def write_agreement_csv(agr: dict, target: str, out: Path):
    rows = []
    cols = [m for m, _ in METHODS]
    for a, b in combinations(cols, 2):
        row = {"target": target,
               "method_a": a, "method_b": b,
               "spearman": float(agr["spearman"].loc[a, b]),
               "pearson":  float(agr["pearson"].loc[a, b])}
        for n in N_LIST:
            row[f"jaccard_top{n}"] = float(agr["jaccard"][n].loc[a, b])
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        out / f"fi_method_agreement_{target}.csv", index=False)


def consensus_top(df: pd.DataFrame, target: str, out: Path, top: int = 20):
    """Average rank across the four tree-based methods."""
    ranks = pd.DataFrame({
        m: df[m].rank(method="average", ascending=False).values
        for m in TREE_METHODS}, index=df.index)
    ranks["mean_tree_rank"] = ranks[TREE_METHODS].mean(axis=1)
    out_df = df[["feature", "resi_i", "resi_j"] + TREE_METHODS].copy()
    for m in TREE_METHODS:
        out_df[f"rank_{m}"] = ranks[m].astype(int)
    out_df["mean_tree_rank"] = ranks["mean_tree_rank"]
    out_df = out_df.sort_values("mean_tree_rank").head(top)
    out_df.to_csv(out / f"fi_consensus_top{top}_{target}.csv", index=False)
    return out_df


def plot_scatter_grid(df: pd.DataFrame, target: str, out: Path):
    """Scatter every method's rank vs SHAP's rank.  Lower rank = more
    important.  Strong agreement = points along y = x."""
    ranks = {m: df[m].rank(method="average", ascending=False).values
             for m, _ in METHODS}
    shap_r = ranks["lgbm_shap_meanabs"]
    others = [m for m in ranks if m != "lgbm_shap_meanabs"]
    labels = dict(METHODS)
    fig, axes = plt.subplots(1, len(others), figsize=(4 * len(others), 3.8),
                             sharex=True, sharey=True)
    for ax, m in zip(axes, others):
        r = ranks[m]
        rho, _ = spearmanr(shap_r, r)
        ax.scatter(shap_r, r, s=6, alpha=0.35, color="#1f77b4",
                   edgecolor="none")
        ax.plot([1, len(shap_r)], [1, len(shap_r)], "k--", lw=0.7,
                alpha=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("SHAP rank")
        ax.set_ylabel(f"{labels[m]} rank")
        ax.set_title(f"ρ = {rho:.2f}")
    fig.suptitle(f"Per-feature rank: SHAP vs other methods, target = {target}",
                 y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fi_method_scatter_{target}.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_topN_jaccard(agrs: dict, out: Path):
    """For each method pair, plot Jaccard@N as a function of N, both
    targets averaged."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = dict(METHODS)
    cols = [m for m, _ in METHODS]
    cmap = plt.get_cmap("tab10")
    pairs = list(combinations(cols, 2))
    for k, (a, b) in enumerate(pairs):
        vals = [(agrs["z0"]["jaccard"][n].loc[a, b]
                 + agrs["z1"]["jaccard"][n].loc[a, b]) / 2
                for n in N_LIST]
        ax.plot(N_LIST, vals, marker="o", lw=1.5, color=cmap(k % 10),
                label=f"{labels[a]} vs {labels[b]}", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Top-N features kept by each method")
    ax.set_ylabel("Jaccard overlap (z0 + z1 averaged)")
    ax.set_title("Top-N pick agreement between FI methods")
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.legend(fontsize=8, ncol=2, loc="upper left",
              bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fi_topN_jaccard.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_consensus_residues(df_z0: pd.DataFrame, df_z1: pd.DataFrame,
                            out: Path, top: int = 50):
    """Per-residue total-rank-product score across both targets and
    four tree methods.  Highlights residues that the consensus picks."""
    def per_residue(df):
        ranks = {m: df[m].rank(method="average", ascending=False)
                 for m in TREE_METHODS}
        df = df.copy()
        df["mean_rank"] = pd.concat(ranks.values(), axis=1).mean(axis=1)

        topdf = df.nsmallest(top, "mean_rank").copy()
        topdf["score"] = 1 / topdf["mean_rank"]
        per_resi = {}
        for _, r in topdf.iterrows():
            for resi in (int(r["resi_i"]), int(r["resi_j"])):
                per_resi[resi] = per_resi.get(resi, 0) + r["score"]
        return per_resi
    r0 = per_residue(df_z0)
    r1 = per_residue(df_z1)
    all_resi = sorted(set(r0) | set(r1))
    x = np.array(all_resi)
    y0 = np.array([r0.get(r, 0) for r in all_resi])
    y1 = np.array([r1.get(r, 0) for r in all_resi])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.4, y0, width=0.8, color="#1976D2", alpha=0.85,
            label="z0 consensus")
    ax.bar(x + 0.4, y1, width=0.8, color="#E65100", alpha=0.85,
            label="z1 consensus")
    ax.set_xlabel("BRAF residue (Kincore numbering)")
    ax.set_ylabel(f"Consensus score (1/mean-rank among top {top})")
    ax.set_title("Per-residue consensus FI across the 4 tree-based methods")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fi_consensus_residues.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended-fi-csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out / "figures"; fig_dir.mkdir(exist_ok=True)

    full = pd.read_csv(args.extended_fi_csv)
    print("Loaded extended FI table:", full.shape)
    print("Available methods:", [c for c in full.columns
                                  if c not in ("target", "feature",
                                                "resi_i", "resi_j")])

    agrs = {}
    consensus = {}
    for t in ("z0", "z1"):
        df = full[full["target"] == t].reset_index(drop=True).copy()
        # Use absolute value for all "score" columns - ridge can be neg.
        for m, _ in METHODS:
            df[m] = df[m].abs()
        print(f"\nTarget {t}: {len(df)} features")
        agrs[t] = agreement_table(df)
        write_agreement_csv(agrs[t], t, args.out)
        consensus[t] = consensus_top(df, t, args.out, top=20)

    plot_topN_jaccard(agrs, fig_dir)
    for t in ("z0", "z1"):
        df = full[full["target"] == t].reset_index(drop=True).copy()
        for m, _ in METHODS:
            df[m] = df[m].abs()
        plot_scatter_grid(df, t, fig_dir)

    df_z0 = full[full["target"] == "z0"].reset_index(drop=True).copy()
    df_z1 = full[full["target"] == "z1"].reset_index(drop=True).copy()
    for d in (df_z0, df_z1):
        for m, _ in METHODS:
            d[m] = d[m].abs()
    plot_consensus_residues(df_z0, df_z1, fig_dir)

    n_features = (full["target"] == "z0").sum()
    # write_html_section(agrs, consensus["z0"], consensus["z1"],
    #                    n_features, args.out)

    # Print headline summary
    print("\n=== HEADLINE SUMMARY ===")
    for t in ("z0", "z1"):
        sp = agrs[t]["spearman"]
        j20 = agrs[t]["jaccard"][20]
        print(f"\nTarget {t}:")
        print(f"  SHAP vs gain   ρ={sp.loc['lgbm_shap_meanabs','lgbm_gain']:.3f}"
              f"  J@20={j20.loc['lgbm_shap_meanabs','lgbm_gain']:.2f}")
        print(f"  SHAP vs perm   ρ={sp.loc['lgbm_shap_meanabs','lgbm_permutation']:.3f}"
              f"  J@20={j20.loc['lgbm_shap_meanabs','lgbm_permutation']:.2f}")
        print(f"  SHAP vs RF     ρ={sp.loc['lgbm_shap_meanabs','rf_impurity']:.3f}"
              f"  J@20={j20.loc['lgbm_shap_meanabs','rf_impurity']:.2f}")
        print(f"  gain vs perm   ρ={sp.loc['lgbm_gain','lgbm_permutation']:.3f}"
              f"  J@20={j20.loc['lgbm_gain','lgbm_permutation']:.2f}")
        print(f"  gain vs RF     ρ={sp.loc['lgbm_gain','rf_impurity']:.3f}"
              f"  J@20={j20.loc['lgbm_gain','rf_impurity']:.2f}")
        print(f"  perm vs RF     ρ={sp.loc['lgbm_permutation','rf_impurity']:.3f}"
              f"  J@20={j20.loc['lgbm_permutation','rf_impurity']:.2f}")


if __name__ == "__main__":
    main()
