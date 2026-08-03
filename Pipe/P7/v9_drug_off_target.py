"""Off-target kinase profiling per FDA inhibitor.

For each verified FDA drug:
  - list every kinase gene it binds in v9 (from the merged full-kinome CSV)
  - classify each kinase as ON-TARGET (FDA-approved primary indication)
    or OFF-TARGET (every other binding observed in PDB)
  - compute per-drug:
      n_on_target_chains, n_off_target_chains
      on-target centroid (z0, z1), off-target centroid
      separation = || on-centroid - off-centroid ||
      off-target dispersion (how spread the off-targets are)
  - flag drugs where off-targets sit in a structurally DIFFERENT
    conformational state from on-targets (potentially relevant to
    off-target toxicity / mechanism)

Outputs:
  per_drug_off_target_table.csv     headline table
  per_drug_off_target_chains.csv    every drug-bound chain with on/off flag
  per_drug_off_target_plot.png/pdf  scatter of drugs with on-target (solid)
                                    + off-target (open) centroids; arrow
                                    from on -> off; labeled by drug
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams


# Primary FDA-approved indications.  This is the set of kinases each
# drug is approved to target (from FDA labels / product monographs).
# Off-target = any v9-bound kinase NOT in this set.
PRIMARY_TARGETS = {
    "STI": {"ABL1", "ABL2", "KIT", "PDGFRA", "PDGFRB"},               # Imatinib
    "NIL": {"ABL1", "ABL2"},                                          # Nilotinib
    "0LI": {"ABL1", "FGFR1", "FGFR2", "FGFR3", "FGFR4",               # Ponatinib
            "KIT", "RET"},
    "1N1": {"ABL1", "SRC", "LCK", "FYN", "LYN", "YES1"},              # Dasatinib (Src+Abl)
    "DB8": {"ABL1", "SRC"},                                           # Bosutinib
    "B49": {"KDR", "FLT1", "FLT3", "FLT4", "KIT", "PDGFRA",           # Sunitinib (multi)
            "PDGFRB", "RET"},
    "BAX": {"BRAF", "RAF1", "ARAF", "KDR", "PDGFRA", "PDGFRB",        # Sorafenib (multi)
            "FLT3", "KIT"},
    "032": {"BRAF"},                                                  # Vemurafenib
    "P06": {"BRAF"},                                                  # Dabrafenib
    "AQ4": {"EGFR"},                                                  # Erlotinib
    "IRE": {"EGFR"},                                                  # Gefitinib
    "FMM": {"EGFR", "ERBB2"},                                         # Lapatinib
    "VGH": {"ALK", "ROS1", "MET"},                                    # Crizotinib
    "0WN": {"EGFR", "ERBB2", "ERBB4"},                                # Afatinib (pan-HER)
    "1E8": {"BTK"},                                                   # Ibrutinib
    "MI1": {"JAK1", "JAK2", "JAK3", "TYK2"},                          # Tofacitinib (pan-JAK)
    "YY3": {"EGFR"},                                                  # Osimertinib (T790M)
    "OZS": {"BTK"},                                                   # Acalabrutinib
    "4MK": {"ALK"},                                                   # Ceritinib
    "LO0": {"NTRK1", "NTRK2", "NTRK3", "ROS1", "ALK"},                # Entrectinib
    "LQQ": {"CDK4", "CDK6"},                                          # Palbociclib
    "TGM": {"MAP2K1", "MAP2K2"},                                      # Trametinib
    "EUI": {"MAP2K1", "MAP2K2"},                                      # Cobimetinib
    "RXT": {"JAK1", "JAK2"},                                          # Ruxolitinib
    # Lab reference and cofactors do not have FDA targets:
    "STU": set(),     # pan-kinase lab reference
    "ANP": set(),     # cofactor
    "ATP": set(),
    "ADP": set(),
}

DRUG_NAMES = {
    "STI": "Imatinib",      "NIL": "Nilotinib",     "0LI": "Ponatinib",
    "1N1": "Dasatinib",     "DB8": "Bosutinib",     "B49": "Sunitinib",
    "BAX": "Sorafenib",     "032": "Vemurafenib",   "P06": "Dabrafenib",
    "AQ4": "Erlotinib",     "IRE": "Gefitinib",     "FMM": "Lapatinib",
    "VGH": "Crizotinib",    "0WN": "Afatinib",      "1E8": "Ibrutinib",
    "MI1": "Tofacitinib",   "YY3": "Osimertinib",   "OZS": "Acalabrutinib",
    "4MK": "Ceritinib",     "LO0": "Entrectinib",   "LQQ": "Palbociclib",
    "TGM": "Trametinib",    "EUI": "Cobimetinib",   "RXT": "Ruxolitinib",
}


def style():
    rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "savefig.dpi": 600,
    })


def parse_kincore_ccd(fasta_path: Path) -> dict[str, set[str]]:
    out = {}
    with fasta_path.open() as f:
        for line in f:
            if not line.startswith(">"):
                continue
            parts = line.rstrip().split("\t")
            ident = parts[0][1:]
            if ident.startswith("AF-"):
                continue
            if len(parts) > 6 and parts[6]:
                raw = parts[6]
                if raw == "No_ligand":
                    out[ident.upper()] = set()
                    continue
                ccds = {e.split(":")[0]
                        for e in raw.split(",") if e and ":" in e}
                ccds = {c for c in ccds if c}
                out[ident.upper()] = ccds
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-kinome-csv", required=True, type=Path)
    ap.add_argument("--kincore-fasta", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()

    df = pd.read_csv(args.full_kinome_csv, keep_default_na=False)
    chain2ccds = parse_kincore_ccd(args.kincore_fasta)
    df["ccd_set"] = df["chain_key"].str.upper().map(
        lambda k: chain2ccds.get(k, set()))
    print(f"v9 full kinome: {len(df)} chains")

    rows = []
    chain_rows = []
    for ccd, on_targets in PRIMARY_TARGETS.items():
        sub = df[df["ccd_set"].apply(lambda s: ccd in s)].copy()
        if len(sub) == 0:
            continue
        name = DRUG_NAMES.get(ccd, ccd)
        sub["on_target"] = sub["gene"].str.upper().isin(on_targets)
        n_on = int(sub["on_target"].sum())
        n_off = len(sub) - n_on
        on = sub[sub["on_target"]]
        off = sub[~sub["on_target"]]
        if len(on):
            on_cent = (on["z0"].mean(), on["z1"].mean())
        else:
            on_cent = (np.nan, np.nan)
        if len(off):
            off_cent = (off["z0"].mean(), off["z1"].mean())
            off_disp = float(np.sqrt(
                ((off[["z0", "z1"]] - np.array(off_cent)) ** 2).sum(axis=1)
                .mean()))
        else:
            off_cent = (np.nan, np.nan)
            off_disp = 0.0
        if len(on) and len(off):
            sep = float(np.hypot(on_cent[0] - off_cent[0],
                                  on_cent[1] - off_cent[1]))
        else:
            sep = np.nan
        rows.append({
            "ccd": ccd, "drug": name,
            "n_on_target_chains":  n_on,
            "n_off_target_chains": n_off,
            "n_on_target_kinases":  on["gene"].nunique(),
            "n_off_target_kinases": off["gene"].nunique(),
            "on_target_kinases":  ";".join(sorted(on["gene"].unique())),
            "off_target_kinases": ";".join(sorted(off["gene"].unique())),
            "on_centroid_z0":  on_cent[0],
            "on_centroid_z1":  on_cent[1],
            "off_centroid_z0": off_cent[0],
            "off_centroid_z1": off_cent[1],
            "off_target_dispersion": off_disp,
            "centroid_separation": sep,
            "primary_targets_declared": ";".join(sorted(on_targets)),
        })
        # Per-chain
        for _, r in sub.iterrows():
            chain_rows.append({
                "drug": name, "ccd": ccd,
                "chain_key": r["chain_key"], "gene": r["gene"],
                "on_target": r["on_target"], "z0": r["z0"], "z1": r["z1"],
                "dfg_spatial": r.get("dfg_spatial", ""),
            })

    summary = pd.DataFrame(rows).sort_values(
        "n_off_target_chains", ascending=False)
    summary.to_csv(args.out_dir / "per_drug_off_target_table.csv",
                    index=False)
    pd.DataFrame(chain_rows).to_csv(
        args.out_dir / "per_drug_off_target_chains.csv", index=False)
    print(f"Wrote per_drug_off_target_table.csv ({len(summary)} drugs)")
    print()
    print("Headline per-drug on/off-target table:")
    print(summary[["drug", "n_on_target_chains", "n_off_target_chains",
                   "n_on_target_kinases", "n_off_target_kinases",
                   "off_target_dispersion", "centroid_separation"]]
          .round(1).to_string(index=False))

    # ---- Figure: on vs off centroids per drug + arrow ----
    # Only plot drugs with both n_on >= 1 and n_off >= 1.
    pl = summary[(summary["n_on_target_chains"] >= 1)
                  & (summary["n_off_target_chains"] >= 1)].copy()
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(df["z0"], df["z1"], s=2, alpha=0.06, color="0.7",
               linewidths=0, zorder=1,
               label=f"all v9 chains (n={len(df):,})")
    cmap = plt.get_cmap("tab20")
    for i, (_, r) in enumerate(pl.iterrows()):
        c = cmap(i % 20)
        # on
        ax.scatter([r["on_centroid_z0"]], [r["on_centroid_z1"]],
                   marker="o", s=110, color=c, edgecolor="black",
                   linewidth=0.8, zorder=4)
        # off (hollow)
        ax.scatter([r["off_centroid_z0"]], [r["off_centroid_z1"]],
                   marker="o", s=110, facecolor="none", edgecolor=c,
                   linewidth=2.5, zorder=4)
        # arrow
        ax.annotate("", xy=(r["off_centroid_z0"], r["off_centroid_z1"]),
                    xytext=(r["on_centroid_z0"], r["on_centroid_z1"]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.5,
                                    alpha=0.75), zorder=3)
        # label drug
        midx = (r["on_centroid_z0"] + r["off_centroid_z0"]) / 2
        midy = (r["on_centroid_z1"] + r["off_centroid_z1"]) / 2
        ax.annotate(f"{r['drug']} ({int(r['n_on_target_chains'])}on,"
                    f" {int(r['n_off_target_chains'])}off)",
                    (midx, midy), fontsize=8, color=c, zorder=5)
    ax.set_xlabel("z0"); ax.set_ylabel("z1")
    ax.set_title("Per-drug on-target (solid) vs off-target (hollow) "
                 "centroid in v9 latent",
                 loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"per_drug_off_target_plot.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote per_drug_off_target_plot.png/pdf "
          f"({len(pl)} drugs with both on- and off-target chains)")


if __name__ == "__main__":
    main()
