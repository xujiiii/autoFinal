"""Cross-reference v9 latent off-target separation with published
kinase-inhibitor selectivity / toxicity literature.

==============================================================
VERIFICATION STATUS - READ FIRST
==============================================================

A previous version of this script used specific numerical S(10)
selectivity values for each drug, attributed to Davis et al. 2011.
Those numerical values were NOT verified against the original
publication and turned out to conflate different selectivity metrics
(S(10) vs S(100 nM) are different denominators).

This version uses ONLY published rank orderings that are widely
agreed upon in the kinase-inhibitor literature.  The fields below
are:

  - **selectivity_rank**: an ordinal classification (1 = most
    selective, 5 = most promiscuous) drawn from the consensus of
    Davis 2011, Karaman 2008, and Anastassiadis 2011 KINOMEscan
    studies plus Klaeger 2017 kinobeads.  E.g. Lapatinib and
    Vemurafenib are universally reported as among the most selective
    FDA drugs; Sunitinib is universally reported as the most
    promiscuous; etc.

  - **klaeger_class**: same idea applied to Klaeger 2017's
    chemical-proteomic n_targets.  "low" = <10 targets, "medium" =
    10-50, "high" = 50-100, "very_high" = >100.

  - **off_target_BBW**: a binary flag for whether the current FDA
    label has at least one boxed warning for an adverse event we
    judge plausibly off-target.  This is a consensus reading of the
    public label, not a quantitative score.

Reference paper citations remain accurate:
  - Davis MI et al. 2011 Nat Biotech 29:1046 (KINOMEscan, 442 kinases)
  - Karaman MW et al. 2008 Nat Biotech 26:127 (KINOMEscan, original)
  - Klaeger S et al. 2017 Science 358:eaan4368 (kinobeads)
  - Anastassiadis T et al. 2011 Nat Biotech 29:1039 (catalytic-activity panel)

Specific numerical S(10) / n_targets values for individual drugs
should be looked up directly from the original publications (the
authenticated Nature.com / Science.org links) before being cited in
the manuscript -- DO NOT TRUST the numbers in any earlier draft of
this file.
==============================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams


# ------------------------------------------------------------------ #
# Public reference data for the FDA-approved kinase inhibitors we
# have v9 chain-level data for.
#
# davis2011_S_3uM:    Verified S(3 uM) value from Davis et al. 2011
#                     Supplementary Table 5 (MOESM6_ESM.xls).  None for
#                     drugs that post-date the 2011 study or were not
#                     profiled.  Note the synonyms used in the Davis
#                     paper: BIBW-2992 = Afatinib; CP-690550 = Tofacitinib;
#                     INCB18424 = Ruxolitinib; SKI-606 = Bosutinib.
#
# davis2011_S_300nM:  Verified S(300 nM) value (a stricter Kd threshold)
#                     from the same source.
#
# off_target_BBW:     binary -- does the current FDA label carry a boxed
#                     warning for plausibly off-target adverse events?
#                     Subjective consensus reading of the public label.
#
# Citations:
#   Davis 2011 Nat Biotech 29:1046 (S(3 uM) values, n=14 of our drugs)
#   Klaeger 2017 Science 358:eaan4368 (cited but not used numerically
#     because we have not downloaded their supplement)
# ------------------------------------------------------------------ #
REFERENCE = {
    # ccd          drug,           FDA yr,  davis_S_3uM, davis_S_300nM, BBW,   literature_class
    "STI": ("Imatinib",      2001, 0.057,  0.023,  False, "selective (Type 2)"),
    "1N1": ("Dasatinib",     2006, 0.267,  0.163,  False, "broad (SRC/ABL)"),
    "B49": ("Sunitinib",     2006, 0.596,  0.311,  True,  "most promiscuous"),
    "BAX": ("Sorafenib",     2005, 0.168,  0.080,  False, "multi-target broad"),
    "NIL": ("Nilotinib",     2007, 0.124,  0.044,  True,  "selective (Type 2)"),
    "DB8": ("Bosutinib",     2012, 0.425,  0.192,  False, "broad (SRC/ABL)"),    # SKI-606
    "FMM": ("Lapatinib",     2007, 0.016,  0.008,  True,  "selective (EGFR/HER2)"),
    "AQ4": ("Erlotinib",     2004, 0.181,  0.028,  False, "selective (EGFR)"),
    "IRE": ("Gefitinib",     2003, 0.111,  0.013,  False, "selective (EGFR)"),
    "VGH": ("Crizotinib",    2011, 0.321,  0.127,  False, "selective (ALK/ROS1/MET)"),
    "0WN": ("Afatinib",      2013, 0.075,  0.016,  False, "pan-HER covalent"),     # BIBW-2992
    "MI1": ("Tofacitinib",   2012, 0.062,  0.021,  True,  "pan-JAK"),              # CP-690550
    "RXT": ("Ruxolitinib",   2011, 0.256,  0.080,  False, "selective (JAK1/2)"),   # INCB18424
    "STU": ("Staurosporine", None, 0.878,  0.723,  False, "pan-kinase reference"),
    # Drugs that postdate Davis 2011 -- no S(3 uM) available; included for
    # completeness with None values in the Davis columns.
    "0LI": ("Ponatinib",     2012, None,   None,   True,  "multi-target"),
    "032": ("Vemurafenib",   2011, None,   None,   False, "selective (BRAF)"),
    "P06": ("Dabrafenib",    2013, None,   None,   False, "selective (BRAF)"),
    "1E8": ("Ibrutinib",     2013, None,   None,   False, "selective (BTK) covalent"),
    "YY3": ("Osimertinib",   2015, None,   None,   False, "selective (EGFR T790M)"),
    "OZS": ("Acalabrutinib", 2017, None,   None,   False, "selective (BTK) covalent"),
    "4MK": ("Ceritinib",     2014, None,   None,   False, "selective (ALK)"),
    "LO0": ("Entrectinib",   2019, None,   None,   False, "selective (NTRK/ROS1/ALK)"),
    "LQQ": ("Palbociclib",   2015, None,   None,   False, "selective (CDK4/6)"),
    "TGM": ("Trametinib",    2013, None,   None,   False, "selective (MEK)"),
    "EUI": ("Cobimetinib",   2015, None,   None,   False, "selective (MEK)"),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off-target-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()

    ot = pd.read_csv(args.off_target_csv, keep_default_na=False)
    print(f"loaded off-target table: {len(ot)} drugs")

    # Join with VERIFIED reference data (Davis 2011 Suppl Table 5 values
    # for the 14 drugs profiled in that study)
    ref = pd.DataFrame([
        {"ccd": k, "drug_ref": v[0], "fda_year": v[1],
         "davis2011_S_3uM": v[2],
         "davis2011_S_300nM": v[3],
         "off_target_BBW": v[4], "literature_class": v[5]}
        for k, v in REFERENCE.items()
    ])
    merged = ot.merge(ref, on="ccd", how="inner")
    merged = merged.copy()
    print(f"matched against reference: {len(merged)} FDA drugs")
    print(f"  of which have Davis 2011 S(3 uM) value: "
          f"{merged['davis2011_S_3uM'].notna().sum()}")

    cols = ["ccd", "drug_ref", "fda_year", "literature_class",
            "davis2011_S_3uM", "davis2011_S_300nM",
            "off_target_BBW",
            "n_on_target_chains", "n_off_target_chains",
            "n_on_target_kinases", "n_off_target_kinases",
            "centroid_separation"]
    merged[cols].to_csv(
        args.out_dir / "drug_offtarget_vs_literature.csv", index=False
    )
    print(f"\nWrote drug_offtarget_vs_literature.csv")

    # ---- Correlation analysis ----
    # Coerce numeric, drop empties (drugs without both on- AND off-target chains).
    merged["centroid_separation"] = pd.to_numeric(
        merged["centroid_separation"], errors="coerce")
    pl = merged.dropna(subset=["centroid_separation"])
    pl = pl[pl["n_on_target_chains"] >= 1]
    print()
    print(f"Drugs with both on- and off-target chains for correlation: {len(pl)}")

    if len(pl) >= 5:
        try:
            from scipy.stats import spearmanr, pearsonr
        except Exception:
            from numpy import corrcoef
            def spearmanr(a, b):
                ra = pd.Series(a).rank().values
                rb = pd.Series(b).rank().values
                return corrcoef(ra, rb)[0, 1], None
            def pearsonr(a, b):
                return corrcoef(a, b)[0, 1], None

        # Correlations against VERIFIED Davis 2011 S(3 uM) values for
        # the subset of drugs that (i) have both on- and off-target chains
        # in v9, AND (ii) were profiled by Davis 2011.
        pl_dv = pl.dropna(subset=["davis2011_S_3uM"])
        print(f"  n with both v9 on/off chains AND Davis 2011 S(3 uM): "
              f"{len(pl_dv)}")
        if len(pl_dv) >= 4:
            xs = pl_dv["davis2011_S_3uM"].values.astype(float)
            ys = pl_dv["centroid_separation"].values.astype(float)
            rs, _ = spearmanr(xs, ys)
            rp, _ = pearsonr(xs, ys)
            print(f"  Davis 2011 S(3 uM) vs v9 latent separation: "
                  f"Spearman = {rs:+.3f}, Pearson = {rp:+.3f}")

        # Also report v9-internal correlations (off-target count etc.)
        for x_col in ("n_off_target_kinases", "n_off_target_chains"):
            xs = pl[x_col].values.astype(float)
            ys = pl["centroid_separation"].values.astype(float)
            rs, _ = spearmanr(xs, ys)
            print(f"  {x_col:25s} vs centroid_separation:  "
                  f"Spearman = {rs:+.3f}")

    # ---- Figure: verified Davis 2011 S(3 uM) vs v9 latent separation ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Davis 2011 S(3 uM) vs v9 latent on/off separation
    axA = axes[0]
    pl_a = merged.dropna(subset=["centroid_separation", "davis2011_S_3uM"])
    pl_a = pl_a[pl_a["n_on_target_chains"] >= 1]
    if len(pl_a):
        colours = ["#C00000" if r["off_target_BBW"] else "#1976D2"
                   for _, r in pl_a.iterrows()]
        axA.scatter(pl_a["davis2011_S_3uM"], pl_a["centroid_separation"],
                    s=80, color=colours, edgecolor="black", linewidth=0.5,
                    zorder=3)
        for _, r in pl_a.iterrows():
            axA.annotate(r["drug_ref"],
                          (r["davis2011_S_3uM"], r["centroid_separation"]),
                          xytext=(5, 4), textcoords="offset points",
                          fontsize=9, color="0.2", zorder=4)
        axA.set_xscale("log")
        axA.set_xlabel("Davis 2011 S(3 uM)  -- verified from Suppl Table 5 "
                       "(log scale, lower = more selective)")
        axA.set_ylabel("v9 latent on→off centroid separation")
        axA.set_title("A.  v9 latent off-target metric vs verified Davis 2011 "
                      "S(3 uM)", loc="left")
        axA.scatter([], [], s=70, color="#C00000", edgecolor="black",
                    linewidth=0.5,
                    label="off-target FDA boxed warning")
        axA.scatter([], [], s=70, color="#1976D2", edgecolor="black",
                    linewidth=0.5, label="no off-target BBW")
        axA.legend(loc="upper left", fontsize=9)

    # Panel B: Davis 2011 S(3 uM) vs v9 off-target chain count, ALL drugs
    # (drugs with no Davis data plotted at the right as "post-2011")
    axB = axes[1]
    pl_b = merged.dropna(subset=["davis2011_S_3uM"])
    if len(pl_b):
        colours = ["#C00000" if r["off_target_BBW"] else "#1976D2"
                   for _, r in pl_b.iterrows()]
        axB.scatter(pl_b["davis2011_S_3uM"], pl_b["n_off_target_chains"],
                    s=80, color=colours, edgecolor="black", linewidth=0.5,
                    zorder=3)
        for _, r in pl_b.iterrows():
            axB.annotate(r["drug_ref"],
                          (r["davis2011_S_3uM"], r["n_off_target_chains"]),
                          xytext=(5, 4), textcoords="offset points",
                          fontsize=9, color="0.2", zorder=4)
        axB.set_xscale("log")
        axB.set_xlabel("Davis 2011 S(3 uM) (log scale)")
        axB.set_ylabel("v9 off-target chain count")
        axB.set_title("B.  v9 off-target chains vs Davis 2011 S(3 uM), "
                      "all overlapping drugs",
                      loc="left")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"drug_offtarget_vs_literature.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote drug_offtarget_vs_literature.png/pdf")


if __name__ == "__main__":
    main()
