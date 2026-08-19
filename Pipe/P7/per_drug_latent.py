"""Per-drug latent-spread analysis on v9.

Pulls the Kincore-derived ligand CCD code per chain, maps a curated
set of FDA-approved kinase inhibitors (plus the lab-standard
staurosporine and ATP cofactors as references) to chain sets, and
computes:

  - per-drug chain count + kinase promiscuity
  - per-drug latent centroid (z0, z1)
  - per-drug latent dispersion (2-D std, RMSD-to-centroid)
  - per-drug DFG-spatial / Kincore-dihedral composition

Outputs:
  per_drug_summary.csv
  per_drug_latent_overlay.{png,pdf}     -- each drug in its own colour
  per_drug_dispersion_bars.{png,pdf}    -- compactness ranked bar chart
  per_drug_class_composition.{png,pdf}  -- stacked DFG-spatial bars

Usage:
  python v9_per_drug_latent.py \
      --latent-csv     manuscript_draft/data/v9_lgbm_shap/v9_latent_with_labels.csv \
      --kincore-fasta  manuscript_draft/data/kincore/PK_labels_PDB.fasta \
      --out-dir        manuscript_draft/data/v9_lgbm_shap/per_drug_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams


# Curated CCD -> (drug name, class, colour) for FDA-approved kinase
# inhibitors with reliable PDB-component-dictionary entries, plus
# staurosporine (the lab pan-kinase reference) and ATP cofactors.
#
# All CCD->drug mappings here have been VERIFIED against the RCSB
# Chemical Component Dictionary REST API (data.rcsb.org/rest/v1/core/chemcomp/{ccd})
# either by exact name match or by IUPAC structure match.  CCDs in the
# project's verification log that did NOT match the expected drug
# (BAY, XAV, TFC, PB2, 9YR, 6S5, 02V, AP5, DJK, EX9, CH7) are excluded.
import re
from typing import Dict, List, Optional
import requests

# 1. 扩展且精准的字典映射（覆盖所有已知激酶抑制剂、辅因子与实验对照）
# 格式: CCD: (Short_Name, Class, Colour, Tier)
KNOWN_DRUG_METADATA = {
    # ---- FDA-approved Type-2 (DFG-out binders) ----
    "STI": ("Imatinib", "Type 2", "#CA0202", "FDA"),
    "NIL": ("Nilotinib", "Type 2", "#FF1C1C", "FDA"),
    "0LI": ("Ponatinib", "Type 2", "#D32F2F", "FDA"),
    "B49": ("Sunitinib", "Type 2", "#E91D3C", "FDA"),
    "BAX": ("Sorafenib", "Type 2", "#FF5050", "FDA"),
    # ---- FDA-approved Type-1.5 (DFG-in, inactive-helix) ----
    "032": ("Vemurafenib", "Type 1.5", "#130EA5", "FDA"),
    "P06": ("Dabrafenib", "Type 1.5", "#00428E", "FDA"),
    # ---- FDA-approved Type-1 & Type-1 Covalent ----
    "1N1": ("Dasatinib", "Type 1", "#1976D2", "FDA"),
    "DB8": ("Bosutinib", "Type 1", "#084197", "FDA"),
    "RXT": ("Ruxolitinib", "Type 1", "#12246D", "FDA"),
    "AQ4": ("Erlotinib", "Type 1", "#0277BD", "FDA"),
    "IRE": ("Gefitinib", "Type 1", "#01579B", "FDA"),
    "FMM": ("Lapatinib", "Type 1", "#283593", "FDA"),
    "VGH": ("Crizotinib", "Type 1", "#512DA8", "FDA"),
    "0WN": ("Afatinib", "Type 1 covalent", "#311B92", "FDA"),
    "1E8": ("Ibrutinib", "Type 1 covalent", "#2335C1", "FDA"),
    "MI1": ("Tofacitinib", "Type 1", "#2A64C8", "FDA"),
    "YY3": ("Osimertinib", "Type 1 covalent", "#0A2E81", "FDA"),
    "OZS": ("Acalabrutinib", "Type 1 covalent", "#0A0C5A", "FDA"),
    "4MK": ("Ceritinib", "Type 1", "#3F51B5", "FDA"),
    "LO0": ("Entrectinib", "Type 1", "#1A237E", "FDA"),
    "LQQ": ("Palbociclib", "Type 1", "#1A046F", "FDA"),
    # ---- FDA-approved Allosteric MEK ----
    "TGM": ("Trametinib", "Allosteric MEK", "#0337E1", "FDA"),
    "EUI": ("Cobimetinib", "Allosteric MEK", "#3A2AB3", "FDA"),
    # ---- Lab reference & Cofactors ----
    "STU": ("Staurosporine", "Type 1 (lab)", "#038D31", "Lab"),
    "ANP": ("AMP-PNP", "cofactor", "#388E3C", "Cofactor"),
    "ATP": ("ATP", "cofactor", "#558B2F", "Cofactor"),
    "ADP": ("ADP", "cofactor", "#8BC34A", "Cofactor"),
}

# 常见盐类、水合物及拉脱维亚/拉丁语后缀的正则表达式（用于文本清洗）
SUFFIX_CLEAN_REGEX = re.compile(
    r"\b(hydrochloride|ditosylate|isethionate|sulfate|mesylate|besylate|maleate|tartrate|succinate|phosphate|sodium|disodium|calcium|potassium|anhydrous|monohydrate|dihydrate|trihydrate|hydrate)\b",
    re.IGNORECASE,
)


def sanitize_drug_name(raw_name: str) -> str:
    """去除药物名中的盐类后缀、-um 拉丁后缀以及多余修饰，保持美观短名"""
    if not raw_name:
        return raw_name

    # 1. 清理常见盐类/水合物单词
    name = SUFFIX_CLEAN_REGEX.sub("", raw_name).strip()

    # 2. 清理 INN 拉丁后缀 -um (如 Vemurafenibum -> Vemurafenib, Crizotinibum -> Crizotinib)
    if name.endswith("um") and len(name) > 5 and not name.lower().endswith("optimum"):
        name = name[:-2]

    # 清理多余空格或末尾标点
    return name.strip(" ,.-")


def extract_clean_drug_name(
    ccd_code: str, chem_name: str, synonyms: list
) -> str:
    """从 API 结果中提取最适合作图展示的药物通用名"""
    candidates = []

    # 收集所有的候选名称
    for syn in synonyms:
        if isinstance(syn, dict):
            s_name = syn.get("name", "")
            p_src = syn.get("provenance_source", "")
            if s_name:
                candidates.append((s_name, p_src))

    # 优先使用来自于 DrugBank 的短名
    for name, src in candidates:
        if src == "DrugBank":
            cleaned = sanitize_drug_name(name)
            if cleaned and len(cleaned) < 30:
                return cleaned.capitalize()

    # 备用选词逻辑：选择最短非全大写的名称
    sorted_candidates = sorted(
        [c[0] for c in candidates], key=lambda x: len(x)
    )
    for c_name in sorted_candidates:
        cleaned = sanitize_drug_name(c_name)
        if cleaned and not cleaned.isupper() and len(cleaned) < 25:
            return cleaned.capitalize()

    # 如果同义词列表没能解析出有效通用名，直接清洗化学全名或保留原 CCD
    cleaned_chem = sanitize_drug_name(chem_name)
    if cleaned_chem and len(cleaned_chem) < 25:
        return cleaned_chem.capitalize()

    return ccd_code


def build_drug_map(ccd_list: List[str]) -> Dict[str, dict]:
    """生成与标准 DRUG_MAP 完全一致的字典结构，未知项目降级为 Other"""
    drug_map = {}

    for ccd in ccd_list:
        ccd_upper = ccd.strip().upper()

        # 优先选择已知库（确保机制分类、颜色和名字100%精准符合论文标准）
        if ccd_upper in KNOWN_DRUG_METADATA:
            name, drug_class, colour, tier = KNOWN_DRUG_METADATA[ccd_upper]
            drug_map[ccd_upper] = {
                "name": name,
                "class": drug_class,
                "colour": colour,
                "tier": tier,
            }
            continue

        # 未知的 CCD 代码 -> 动态请求 RCSB REST API
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ccd_upper}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                # 无法在 RCSB 查询到的项目归为 Other
                drug_map[ccd_upper] = {
                    "name": ccd_upper,
                    "class": "Other",
                    "colour": "#757575",
                    "tier": "Unknown",
                }
                continue

            data = resp.json()
            chem_name = data.get("chem_comp", {}).get("name", "")
            synonyms = data.get("rcsb_chem_comp_synonyms", [])

            # 提取并清洗名字
            drug_name = extract_clean_drug_name(
                ccd_upper, chem_name, synonyms
            )

            # 未知具体构象/机制分类的药物，归为 Other，颜色给中性灰 #757575
            drug_map[ccd_upper] = {
                "name": drug_name,
                "class": "Other",
                "colour": "#757575",
                "tier": "Other",
            }

        except Exception as e:
            drug_map[ccd_upper] = {
                "name": ccd_upper,
                "class": "Other",
                "colour": "#757575",
                "tier": "Unknown",
            }

    return drug_map

ccds=['STI', 'NIL', '0LI', 'B49', 'BAX', '032', 'P06',
        '1N1', 'DB8', 'RXT', 'AQ4', 'IRE', 'FMM', 
        'VGH', '0WN', '1E8', 'MI1', 'YY3', 'OZS', '4MK',
        'LO0', 'LQQ', 'TGM', 'EUI', 'STU', 'ANP', 'ATP', 'ADP']

DRUG_MAP = build_drug_map(ccds)


# DRUG_MAP: dict[str, dict] = {
#     # ---- FDA-approved Type-2 (DFG-out binders) ----
#     "STI": {"name": "Imatinib",     "class": "Type 2",
#             "colour": "#C00000", "tier": "FDA"},
#     "NIL": {"name": "Nilotinib",    "class": "Type 2",
#             "colour": "#B71C1C", "tier": "FDA"},
#     "0LI": {"name": "Ponatinib",    "class": "Type 2",
#             "colour": "#D32F2F", "tier": "FDA"},
#     "B49": {"name": "Sunitinib",    "class": "Type 2",
#             "colour": "#9C27B0", "tier": "FDA"},
#     "BAX": {"name": "Sorafenib",    "class": "Type 2",
#             "colour": "#E65100", "tier": "FDA"},

#     # ---- FDA-approved Type-1.5 (DFG-in but inactive-helix-shifted) ----
#     "032": {"name": "Vemurafenib",  "class": "Type 1.5",
#             "colour": "#5D4037", "tier": "FDA"},
#     "P06": {"name": "Dabrafenib",   "class": "Type 1.5",
#             "colour": "#795548", "tier": "FDA"},

#     # ---- FDA-approved Type-1 (DFG-in binders) ----
#     "1N1": {"name": "Dasatinib",    "class": "Type 1",
#             "colour": "#1976D2", "tier": "FDA"},
#     "DB8": {"name": "Bosutinib",    "class": "Type 1",
#             "colour": "#00838F", "tier": "FDA"},
#     "RXT": {"name": "Ruxolitinib",  "class": "Type 1",
#             "colour": "#FF9800", "tier": "FDA"},
#     "AQ4": {"name": "Erlotinib",    "class": "Type 1",
#             "colour": "#0277BD", "tier": "FDA"},
#     "IRE": {"name": "Gefitinib",    "class": "Type 1",
#             "colour": "#01579B", "tier": "FDA"},
#     "FMM": {"name": "Lapatinib",    "class": "Type 1",
#             "colour": "#283593", "tier": "FDA"},
#     "VGH": {"name": "Crizotinib",   "class": "Type 1",
#             "colour": "#512DA8", "tier": "FDA"},
#     "0WN": {"name": "Afatinib",     "class": "Type 1 covalent",
#             "colour": "#311B92", "tier": "FDA"},
#     "1E8": {"name": "Ibrutinib",    "class": "Type 1 covalent",
#             "colour": "#4A148C", "tier": "FDA"},
#     "MI1": {"name": "Tofacitinib",  "class": "Type 1",
#             "colour": "#006064", "tier": "FDA"},
#     "YY3": {"name": "Osimertinib",  "class": "Type 1 covalent",
#             "colour": "#37474F", "tier": "FDA"},
#     "OZS": {"name": "Acalabrutinib", "class": "Type 1 covalent",
#             "colour": "#263238", "tier": "FDA"},
#     "4MK": {"name": "Ceritinib",    "class": "Type 1",
#             "colour": "#3F51B5", "tier": "FDA"},
#     "LO0": {"name": "Entrectinib",  "class": "Type 1",
#             "colour": "#1A237E", "tier": "FDA"},
#     "LQQ": {"name": "Palbociclib",  "class": "Type 1",
#             "colour": "#FF5722", "tier": "FDA"},

#     # ---- FDA-approved Allosteric (MEK1/2 inhibitors) ----
#     "TGM": {"name": "Trametinib",   "class": "Allosteric MEK",
#             "colour": "#F06292", "tier": "FDA"},
#     "EUI": {"name": "Cobimetinib",  "class": "Allosteric MEK",
#             "colour": "#EC407A", "tier": "FDA"},

#     # ---- Lab pan-kinase reference ----
#     "STU": {"name": "Staurosporine", "class": "Type 1 (lab)",
#             "colour": "#7B1FA2", "tier": "Lab"},

#     # ---- Cofactor reference ----
#     "ANP": {"name": "AMP-PNP",       "class": "cofactor",
#             "colour": "#388E3C", "tier": "Cofactor"},
#     "ATP": {"name": "ATP",           "class": "cofactor",
#             "colour": "#558B2F", "tier": "Cofactor"},
#     "ADP": {"name": "ADP",           "class": "cofactor",
#             "colour": "#8BC34A", "tier": "Cofactor"},
# }


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


def parse_kincore_ccd(fasta_path: Path) -> dict[str, set[str]]:
    """chain_key (upper) -> set of CCDs of all bound ligands.

    Kincore stores the ligand column as ``CCD:authnum`` for single
    ligands and ``CCD1:a,CCD2:b,...`` for chains with multiple bound
    ligands.  We return the *set* of distinct CCDs per chain so a
    chain bound to both, say, imatinib and ATP appears in both
    drug analyses below.
    """
    out: dict[str, set[str]] = {}
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
                ccds = set()
                for entry in raw.split(","):
                    entry = entry.strip()
                    if not entry:
                        continue
                    ccd = entry.split(":")[0]
                    if ccd:
                        ccds.add(ccd)
                out[ident.upper()] = ccds
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-csv", required=True, type=Path)
    ap.add_argument("--kincore-fasta", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--addendum-csv", type=Path, default=None,
                    help="Optional CSV of OOD-encoded chains (e.g. EGFR/ALK "
                         "from extract_and_encode_addendum.py) to merge into "
                         "the static-PDB latent.  These chains were NOT in "
                         "v9 training; they are marked ood=True in the per-"
                         "drug summary and plotted with hollow markers.")
    ap.add_argument("--min-n-for-plot", type=int, default=3,
                    help="Drugs with fewer chains are excluded from figures "
                         "but kept in the summary CSV.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()

    df = pd.read_csv(args.latent_csv, keep_default_na=False)
    df["ood"] = False
    chain2ccds = parse_kincore_ccd(args.kincore_fasta)
    df["ccd_set"] = df["chain_key"].str.upper().map(
        lambda k: chain2ccds.get(k, set()))
    df["ccd_count"] = df["ccd_set"].apply(len)
    print(f"v9 chains (in-distribution): {len(df)}, "
          f"with any CCD: {(df['ccd_count'] > 0).sum()}, "
          f"with multi-ligand: {(df['ccd_count'] > 1).sum()}")

    # ---- merge addendum (out-of-distribution chains) ----
    if args.addendum_csv is not None:
        add = pd.read_csv(args.addendum_csv, keep_default_na=False)
        # The addendum carries its own ligand parsing because the kincore-
        # fasta lookup keyed by chain_key already works.
        add["ccd_set"] = add["chain_key"].str.upper().map(
            lambda k: chain2ccds.get(k, set()))
        add["ccd_count"] = add["ccd_set"].apply(len)
        add["ood"] = True
        # align columns to df.
        for c in df.columns:
            if c not in add.columns:
                add[c] = ""
        df = pd.concat([df, add[df.columns]], ignore_index=True)
        print(f"+ addendum (OOD-encoded) chains: {len(add)} "
              f"(genes: {sorted(add['gene'].unique().tolist())})")
        print(f"total chains after merge: {len(df)}")

    def chains_with_ccd(ccd: str) -> pd.DataFrame:
        """Chains that have ``ccd`` among their bound ligands."""
        mask = df["ccd_set"].apply(lambda s: ccd in s)
        return df[mask]

    # ---- per-drug summary ----
    rows = []
    for ccd, info in DRUG_MAP.items():
        sub = chains_with_ccd(ccd)
        if len(sub) == 0:
            continue
        z = sub[["z0", "z1"]].values
        cent = z.mean(axis=0)
        # 2-D dispersion: mean Euclidean distance from centroid (RMSD-style)
        dispersion = float(np.sqrt(((z - cent) ** 2).sum(axis=1).mean()))
        # Component stds
        s0 = float(z[:, 0].std()); s1 = float(z[:, 1].std())
        spatial_counts = sub["dfg_spatial"].value_counts().to_dict()
        dihedral_counts = sub["dihedral"].value_counts().to_dict()
        n_ood = int(sub["ood"].sum()) if "ood" in sub.columns else 0
        rows.append({
            "ccd":          ccd,
            "name":         info["name"],
            "drug_class":   info["class"],
            "tier":         info["tier"],
            "n_chains":     len(sub),
            "n_ood_chains": n_ood,
            "n_kinases":    sub["gene"].nunique(),
            "z0_centroid":  float(cent[0]),
            "z1_centroid":  float(cent[1]),
            "z0_std":       s0,
            "z1_std":       s1,
            "dispersion":   dispersion,
            "frac_DFGin":   spatial_counts.get("DFGin", 0) / len(sub),
            "frac_DFGout":  spatial_counts.get("DFGout", 0) / len(sub),
            "frac_DFGinter": spatial_counts.get("DFGinter", 0) / len(sub),
            "frac_BLAminus": dihedral_counts.get("BLAminus", 0) / len(sub),
            "frac_BLBplus":  dihedral_counts.get("BLBplus", 0) / len(sub),
            "frac_BLBminus": dihedral_counts.get("BLBminus", 0) / len(sub),
            "frac_BBAminus": dihedral_counts.get("BBAminus", 0) / len(sub),
            "kinase_list":   ";".join(sorted(sub["gene"].unique())),
        })
    summary = pd.DataFrame(rows).sort_values("dispersion")
    summary.to_csv(args.out_dir / "per_drug_summary.csv", index=False)
    print(f"\nWrote per_drug_summary.csv ({len(summary)} drugs)")
    print(summary[["ccd", "name", "n_chains", "n_kinases",
                   "z0_centroid", "z1_centroid", "dispersion",
                   "frac_DFGin", "frac_DFGout"]].to_string(index=False))

    # For plotting, drop drugs with too few chains for meaningful stats.
    summary_plot = summary[summary["n_chains"] >= args.min_n_for_plot].copy()
    print(f"\nDrugs with n>={args.min_n_for_plot} chains "
          f"(used in figures): {len(summary_plot)}/{len(summary)}")

    # ---- Figure 1: per-drug overlay on latent ----
    fig, ax = plt.subplots(figsize=(9, 8))
    # All v9 chains as light grey background
    ax.scatter(df["z0"], df["z1"], s=4, alpha=0.10,
               color="0.7", linewidths=0, zorder=1,
               label=f"all chains (n={len(df):,})")
    # Draw drugs in the curated order: cofactors first (back), drugs front.
    order = sorted(summary_plot["ccd"].tolist(),
                   key=lambda c: 0 if DRUG_MAP[c]["tier"] == "Cofactor"
                                 else 1 if DRUG_MAP[c]["tier"] == "Lab"
                                 else 2)
    for ccd in order:
        sub = chains_with_ccd(ccd)
        if len(sub) == 0:
            continue
        info = DRUG_MAP[ccd]
        marker = "s" if info["tier"] == "Cofactor" else \
                 "^" if info["tier"] == "Lab" else "o"
        size = 28 if info["tier"] == "Cofactor" else 60
        alpha = 0.5 if info["tier"] == "Cofactor" else 0.85
        n_total = len(sub)
        n_ood = int(sub["ood"].sum()) if "ood" in sub.columns else 0
        sub_id = sub[~sub["ood"]] if "ood" in sub.columns else sub
        sub_ood = sub[sub["ood"]] if "ood" in sub.columns else sub.iloc[:0]
        if len(sub_id):
            ax.scatter(sub_id["z0"], sub_id["z1"], s=size, alpha=alpha,
                       color=info["colour"], marker=marker,
                       edgecolor="black", linewidth=0.4, zorder=3)
        if len(sub_ood):
            # OOD chains (EGFR/ALK projected without training) get a hollow
            # marker with dashed edge so they stand out as provisional.
            ax.scatter(sub_ood["z0"], sub_ood["z1"], s=size + 25,
                       facecolor="none", edgecolor=info["colour"],
                       linewidth=1.8, marker=marker, zorder=4)
        suffix = (f", n={n_total}" if n_ood == 0
                  else f", n={n_total} ({n_ood} OOD)")
        ax.scatter([], [], s=size, color=info["colour"], marker=marker,
                   edgecolor="black", linewidth=0.4, alpha=alpha,
                   label=f"{info['name']} ({ccd}{suffix}, {info['class']})")
        c = (sub["z0"].mean(), sub["z1"].mean())
        ax.scatter([c[0]], [c[1]], marker="X", s=140,
                   color=info["colour"], edgecolor="white",
                   linewidth=1.5, zorder=5)
    ax.set_xlabel("z0")
    ax.set_ylabel("z1")
    ax.set_title("Per-drug latent footprints on latent space", loc="left")
    ax.legend(loc="upper left", fontsize=8, markerscale=0.9,
              bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"per_drug_latent_overlay.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote per_drug_latent_overlay.png/pdf")

    # ---- Figure 2: dispersion bar chart ----
    fig, ax = plt.subplots(figsize=(8, 5))
    summary_sorted = summary_plot.sort_values("dispersion", ascending=True)
    colours = [DRUG_MAP[c]["colour"] for c in summary_sorted["ccd"]]
    labels = [f"{r['name']} ({r['ccd']})" for _, r in summary_sorted.iterrows()]
    ax.barh(range(len(summary_sorted)), summary_sorted["dispersion"],
            color=colours, edgecolor="black", linewidth=0.4)
    for i, (_, r) in enumerate(summary_sorted.iterrows()):
        ax.text(r["dispersion"] + 1.0, i,
                f" n={r['n_chains']}, {r['n_kinases']} kinases",
                va="center", fontsize=9, color="0.25")
    ax.set_yticks(range(len(summary_sorted)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(r"latent dispersion (mean Euclidean distance to centroid)")
    ax.set_title("Per-drug compactness in latent (lower = tighter)",
                 loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"per_drug_dispersion_bars.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote per_drug_dispersion_bars.png/pdf")

    # ---- Figure 3: DFG-spatial composition stacked bars ----
    fig, ax = plt.subplots(figsize=(8, 5))
    spatial_cols = ["frac_DFGin", "frac_DFGinter", "frac_DFGout"]
    spatial_colors = ["#9CB7E0", "#E8C26F", "#D67A7A"]
    spatial_labels = ["DFGin", "DFGinter", "DFGout"]
    summary_by_class = summary_plot.sort_values(
        ["drug_class", "frac_DFGout"], ascending=[True, False]
    )
    y = np.arange(len(summary_by_class))
    left = np.zeros(len(summary_by_class))
    for col, colr, lbl in zip(spatial_cols, spatial_colors, spatial_labels):
        ax.barh(y, summary_by_class[col].values, left=left,
                color=colr, edgecolor="black", linewidth=0.4,
                label=lbl)
        left = left + summary_by_class[col].values
    labels = [f"{r['name']} ({r['ccd']}, {r['drug_class']})"
              for _, r in summary_by_class.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("fraction of chains")
    ax.set_xlim(0, 1.0)
    ax.set_title("Per-drug DFG-spatial composition", loc="left")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"per_drug_class_composition.{ext}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote per_drug_class_composition.png/pdf")


if __name__ == "__main__":
    main()
