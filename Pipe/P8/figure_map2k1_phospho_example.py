"""Focused figure for the MAP2K1 phospho-site worked example.

Two panels:
  A. Global v9 latent (all 5,318 chains) coloured by Kincore DFG-spatial
     class, with MAP2K1 WT / S218A+S222A / S218D+S222D overlaid as
     highlighted points + centroids.
  B. Zoom-out arrow plot: WT centroid -> phospho-dead centroid,
     WT centroid -> phospho-mimetic centroid, labelled by displacement.

Reads:
  manuscript_draft/data/v9_lgbm_shap/v9_latent_with_labels.csv

Output:
  <out-prefix>.png  +  <out-prefix>.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams


S218A_S222A_KEYS = (
    "6NYBB;6PP9B;6Q0JC;6Q0JD;6Q0TC;6V2WB;7M0TB;7M0UB;7M0VB;"
    "7M0WB;7M0XB;7M0YB;7M0ZB;8CHFE;8CHFF;8DGSB;8DGTB"
).split(";")
S218D_S222D_KEYS = ["5YT3B", "5YT3D"]


DFG_COLORS = {
    "DFGin":    "#9CB7E0",
    "DFGinter": "#E8C26F",
    "DFGout":   "#D67A7A",
    "":         "#D0D0D0",
    "noise":    "#D0D0D0",
}


def style():
    rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "savefig.dpi": 600,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-csv", required=True, type=Path)
    ap.add_argument("--out-prefix", required=True, type=Path)
    args = ap.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    style()

    df = pd.read_csv(args.latent_csv, keep_default_na=False)
    print(f"Loaded {len(df)} chains")

    mek = df[df["gene"].str.upper() == "MAP2K1"].copy()
    mek["mut_label"] = "WT / other"
    mek.loc[mek["chain_key"].isin(S218A_S222A_KEYS),
            "mut_label"] = "S218A+S222A (phospho-dead)"
    mek.loc[mek["chain_key"].isin(S218D_S222D_KEYS),
            "mut_label"] = "S218D+S222D (phospho-mimetic)"
    print(f"MAP2K1 chains: {len(mek)}")
    print(mek["mut_label"].value_counts().to_string())

    centroids = mek.groupby("mut_label")[["z0", "z1"]].mean()
    print(); print(centroids)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---- Panel A: full latent coloured by DFG-spatial + MAP2K1 overlay ----
    axA = axes[0]
    for spatial in ("DFGin", "DFGinter", "DFGout"):
        sub = df[df["dfg_spatial"] == spatial]
        axA.scatter(sub["z0"], sub["z1"], s=4, alpha=0.30,
                    color=DFG_COLORS.get(spatial, "#D0D0D0"),
                    linewidths=0, zorder=1,
                    label=f"{spatial} (n={len(sub)})")
    # un/none-labelled chains
    others = df[~df["dfg_spatial"].isin({"DFGin", "DFGinter", "DFGout"})]
    if len(others):
        axA.scatter(others["z0"], others["z1"], s=4, alpha=0.20,
                    color="#D0D0D0", linewidths=0, zorder=1,
                    label=f"unlabelled (n={len(others)})")

    # MAP2K1 WT chains in darker grey
    wt = mek[mek["mut_label"] == "WT / other"]
    axA.scatter(wt["z0"], wt["z1"], s=30, alpha=0.85,
                color="#4D4D4D", edgecolor="black", linewidth=0.4,
                zorder=3, label=f"MAP2K1 WT / other (n={len(wt)})")
    # phospho-dead
    pd_ = mek[mek["mut_label"] == "S218A+S222A (phospho-dead)"]
    axA.scatter(pd_["z0"], pd_["z1"], s=70, alpha=0.95,
                color="#3C78D8", marker="o",
                edgecolor="black", linewidth=0.5, zorder=4,
                label=f"S218A+S222A (n={len(pd_)})")
    # phospho-mimetic
    pm = mek[mek["mut_label"] == "S218D+S222D (phospho-mimetic)"]
    axA.scatter(pm["z0"], pm["z1"], s=140, alpha=0.95,
                color="#CC0000", marker="*",
                edgecolor="black", linewidth=0.6, zorder=5,
                label=f"S218D+S222D (n={len(pm)})")
    axA.set_xlabel("z0")
    axA.set_ylabel("z1")
    axA.set_title("A.  global v9 latent, MAP2K1 chains highlighted",
                  loc="left")
    axA.legend(loc="lower right", fontsize=9, markerscale=1.4)

    # ---- Panel B: per-chain MAP2K1 with arrows ----
    axB = axes[1]
    axB.scatter(wt["z0"], wt["z1"], s=45, alpha=0.85,
                color="#4D4D4D", edgecolor="black", linewidth=0.4,
                zorder=3, label=f"MAP2K1 WT / other (n={len(wt)})")
    axB.scatter(pd_["z0"], pd_["z1"], s=80, alpha=0.95,
                color="#3C78D8", marker="o",
                edgecolor="black", linewidth=0.5, zorder=4,
                label=f"S218A+S222A (n={len(pd_)})")
    axB.scatter(pm["z0"], pm["z1"], s=180, alpha=0.95,
                color="#CC0000", marker="*",
                edgecolor="black", linewidth=0.6, zorder=5,
                label=f"S218D+S222D (n={len(pm)})")

    wt_c = centroids.loc["WT / other"]
    pd_c = centroids.loc["S218A+S222A (phospho-dead)"]
    pm_c = centroids.loc["S218D+S222D (phospho-mimetic)"]

    # WT centroid
    axB.scatter([wt_c["z0"]], [wt_c["z1"]], marker="X", s=200,
                color="#101010", edgecolor="white", linewidth=1.5,
                zorder=6)
    axB.annotate("WT centroid", (wt_c["z0"], wt_c["z1"]),
                 xytext=(10, 8), textcoords="offset points",
                 fontsize=10, color="#101010", zorder=7)

    # Arrows
    axB.annotate(
        "", xy=(pd_c["z0"], pd_c["z1"]),
        xytext=(wt_c["z0"], wt_c["z1"]),
        arrowprops=dict(arrowstyle="->", color="#3C78D8",
                        lw=2.0, alpha=0.85), zorder=6,
    )
    axB.annotate(
        "", xy=(pm_c["z0"], pm_c["z1"]),
        xytext=(wt_c["z0"], wt_c["z1"]),
        arrowprops=dict(arrowstyle="->", color="#CC0000",
                        lw=2.0, alpha=0.85), zorder=6,
    )

    # Distance annotations
    dpd = np.hypot(pd_c["z0"] - wt_c["z0"], pd_c["z1"] - wt_c["z1"])
    dpm = np.hypot(pm_c["z0"] - wt_c["z0"], pm_c["z1"] - wt_c["z1"])
    midA = ((wt_c["z0"] + pd_c["z0"]) / 2, (wt_c["z1"] + pd_c["z1"]) / 2)
    midB = ((wt_c["z0"] + pm_c["z0"]) / 2, (wt_c["z1"] + pm_c["z1"]) / 2)
    axB.text(midA[0] - 5, midA[1] + 3,
             f"|Δ| = {dpd:.1f}\nMahal σ = 1.94\nperm p = 4×10⁻⁴",
             fontsize=10, color="#3C78D8", ha="right", va="bottom")
    axB.text(midB[0] + 3, midB[1] + 5,
             f"|Δ| = {dpm:.1f}\nMahal σ = 16.4\nperm p = 1.2×10⁻³",
             fontsize=10, color="#CC0000", ha="left", va="bottom")

    axB.set_xlabel("z0")
    axB.set_ylabel("z1")
    axB.set_title("B.  WT → mutant displacement in the v9 latent",
                  loc="left")
    axB.legend(loc="lower right", fontsize=9, markerscale=1.0)

    fig.tight_layout()
    out_png = str(args.out_prefix) + ".png"
    out_pdf = str(args.out_prefix) + ".pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
