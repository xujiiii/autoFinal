"""Four v9 latent analyses run together:

  1. Within-kinase conformational diversity: a multi-panel scatter
     showing how individual high-data-coverage kinases populate
     distinct latent regions.
  2. Novel-region discovery: cluster Kincore-None chains (no dihedral
     label) in the v9 latent; identify clusters that look like
     candidate new conformational classes; decode each cluster's
     latent centroid back to a 27-Cα structure.
  3. Nearest-neighbour structural validation: for a random sample,
     compare per-chain Cα-RMSD to the 5 latent-nearest chains vs the
     5 random chains. If the latent encodes real structural similarity,
     latent neighbours should be much closer.
  4. ABL1 escape-route case study: map ABL1's resistance mutations
     (T315I, T315A, V299L, F317L, F359V, F359C, M388L, Y253H, E255K,
     G250E, etc.) in latent space; do they share an escape direction?

Outputs (under <out>/):
  within_kinase_diversity.png + .csv
  novel_regions/latent_clusters.png + decoded_centroids.pdb + summary.csv
  nn_validation.png + nn_validation.csv
  abl1_escape_routes.png + abl1_escape_routes.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

DFG_COLORS = {
    "DFGin": "#4c78a8", "DFGout": "#f58518",
    "DFGinter": "#54a24b", "None": "#bbbbbb",
}
DIH_COLORS = {
    "BLAminus": "#4c78a8", "BLAplus": "#72b7b2",
    "ABAminus": "#f58518", "BLBminus": "#54a24b",
    "BLBplus": "#b79a20", "BLBtrans": "#e45756",
    "BBAminus": "#b279a2", "BABtrans": "#ff9da6",
    "None": "#bbbbbb",
}
LIG_COLORS = {
    "Type1": "#4c78a8", "Type2": "#f58518",
    "Type1.5_Back": "#9c6dc3", "Type1.5_Front": "#a072cd",
    "Type3": "#e45756", "Allosteric": "#54a24b",
    "ATPlike": "#b79a20", "No_ligand": "#bbbbbb",
}


def load_combined_pdb(path: Path) -> np.ndarray:
    out, cur = [], []
    with path.open() as f:
        for line in f:
            if line.startswith("MODEL"):
                cur = []
            elif line.startswith("ATOM") and line[12:16].strip() == "CA":
                cur.append([float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54])])
            elif line.startswith("ENDMDL") and cur:
                out.append(cur); cur = []
    return np.asarray(out, dtype=np.float32)


def per_chain_rmsd_pairs(a, b):
    return float(np.sqrt(((a - b)**2).sum(axis=-1).mean()))


# -------------------- (1) Within-kinase diversity ----------------------------


def analysis_within_kinase(df, out_dir, target_genes=None):
    print("\n=== (1) Within-kinase diversity ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    if target_genes is None:
        # Pick the genes with most chains AND notable mention in kinase literature.
        candidates = ["ABL1", "EGFR", "BRAF", "CDK2", "MAPK14", "AURKA",
                      "MAPK1", "KIT", "JAK2", "FGFR1", "MAP2K1", "PRKACA"]
        counts = df["gene"].value_counts().to_dict()
        target_genes = [g for g in candidates if counts.get(g, 0) >= 20][:8]
    print(f"  Genes plotted: {target_genes}")

    ncol = 4
    nrow = (len(target_genes) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 4.5*nrow),
                              squeeze=False, sharex=True, sharey=True)

    for k, gene in enumerate(target_genes):
        ax = axes[k // ncol, k % ncol]
        # background
        ax.scatter(df["z0"], df["z1"], s=2, alpha=0.1,
                   color="#dddddd", edgecolor="none")
        sub = df[df["gene"] == gene]
        for cls, color in DFG_COLORS.items():
            m = sub["dfg_spatial"] == cls
            if m.sum():
                ax.scatter(sub.loc[m, "z0"], sub.loc[m, "z1"],
                           s=16, alpha=0.85, color=color,
                           label=f"{cls} ({int(m.sum())})",
                           edgecolor="none")
        ax.set_title(f"{gene} (n={len(sub)})", fontsize=12)
        ax.legend(fontsize=8, frameon=False, loc="best")
    for k in range(len(target_genes), nrow*ncol):
        axes[k // ncol, k % ncol].axis("off")

    fig.suptitle("(1) Within-kinase conformational diversity in v9 latent\n"
                 "background: all chains in grey; each kinase coloured by Kincore DFG state",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "within_kinase_diversity.png")
    plt.close(fig)

    # CSV summary: per-gene #latent-clusters via simple KMeans
    from sklearn.cluster import KMeans
    rows = []
    for gene in target_genes:
        sub = df[df["gene"] == gene]
        if len(sub) < 5: continue
        Z = sub[["z0", "z1"]].to_numpy()
        for k_try in [2, 3, 4]:
            if len(sub) < k_try*3: continue
            km = KMeans(n_clusters=k_try, random_state=25,
                        n_init=10).fit(Z)
            labels = km.labels_
            sizes = pd.Series(labels).value_counts().sort_index().tolist()
            rows.append({"gene": gene, "n_chains": len(sub),
                         "k_means_k": k_try,
                         "cluster_sizes": ";".join(map(str, sizes)),
                         "kmeans_inertia": float(km.inertia_)})
    pd.DataFrame(rows).to_csv(out_dir / "within_kinase_diversity.csv",
                              index=False)
    print(f"  wrote within_kinase_diversity.{{png,csv}}")


# -------------------- (2) Novel-region discovery -----------------------------


def analysis_novel_regions(df, out_dir, coords_path, checkpoint, n_atoms=27):
    print("\n=== (2) Novel-region discovery (Kincore-None chains) ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    none_mask = df["dihedral"].astype(str).isin(["", "None", "nan"])
    print(f"  Kincore-None chains: {none_mask.sum()}")
    if none_mask.sum() < 10:
        print("  Not enough None chains; skipping.")
        return

    none_df = df[none_mask].copy()
    Z = none_df[["z0", "z1"]].to_numpy()

    # KMeans + silhouette to pick k
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best_k = 2; best_s = -1
    for k_try in range(2, 9):
        if len(Z) < k_try*3: continue
        km = KMeans(n_clusters=k_try, random_state=25, n_init=10).fit(Z)
        if len(set(km.labels_)) < 2: continue
        s = silhouette_score(Z, km.labels_)
        if s > best_s:
            best_s = s; best_k = k_try
    print(f"  Best k by silhouette: k={best_k} (silhouette={best_s:.3f})")
    km = KMeans(n_clusters=best_k, random_state=25, n_init=10).fit(Z)
    none_df["cluster"] = km.labels_

    # Cluster summary
    cluster_rows = []
    for c in range(best_k):
        sub = none_df[none_df["cluster"] == c]
        cluster_rows.append({
            "cluster": int(c),
            "n_chains": int(len(sub)),
            "centroid_z0": float(km.cluster_centers_[c][0]),
            "centroid_z1": float(km.cluster_centers_[c][1]),
            "top_genes": ";".join(sub["gene"].value_counts().head(5).index.tolist()),
            "top_families": ";".join(
                sub["group"].astype(str).str.split("_").str[0]
                .value_counts().head(3).index.tolist()),
            "dfg_in_frac": float((sub["dfg_spatial"]=="DFGin").mean()),
            "dfg_out_frac": float((sub["dfg_spatial"]=="DFGout").mean()),
        })
    pd.DataFrame(cluster_rows).to_csv(
        out_dir / "novel_region_clusters.csv", index=False)
    print(pd.DataFrame(cluster_rows).to_string(index=False))

    # Decode centroids if checkpoint is provided
    decoded_centroids = {}
    if checkpoint and checkpoint.exists():
        from molearn.models.small_foldingnet import Small_AutoEncoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = Small_AutoEncoder(out_points=n_atoms).to(device)
        state = torch.load(str(checkpoint), map_location=device,
                           weights_only=False)
        sd = state.get("model_state_dict", state)
        net.load_state_dict(sd); net.eval()
        with torch.no_grad():
            for c in range(best_k):
                z = torch.tensor(km.cluster_centers_[c], dtype=torch.float32,
                                  device=device).unsqueeze(0)
                out = net.decoder(z)
                if out.shape[1] == 3 and out.shape[-1] >= n_atoms:
                    out = out.permute(0, 2, 1)
                pts = out[0, :n_atoms].cpu().numpy()
                decoded_centroids[c] = pts
        # Write a multi-MODEL PDB of decoded centroids
        with (out_dir / "decoded_centroids.pdb").open("w") as f:
            for c, pts in decoded_centroids.items():
                f.write(f"MODEL {c}\n")
                for i, p in enumerate(pts, start=1):
                    f.write(f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
                            f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}"
                            "  1.00  0.00           C\n")
                f.write("ENDMDL\n")
            f.write("END\n")
        print(f"  wrote decoded_centroids.pdb ({len(decoded_centroids)} models)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    # Left: all v9 chains in grey + Kincore-None coloured by cluster
    ax = axes[0]
    ax.scatter(df["z0"], df["z1"], s=2, alpha=0.1, color="#dddddd",
               edgecolor="none")
    cluster_colors = plt.cm.tab10.colors
    for c in range(best_k):
        sub = none_df[none_df["cluster"] == c]
        ax.scatter(sub["z0"], sub["z1"], s=20, alpha=0.8,
                   color=cluster_colors[c % 10],
                   label=f"cluster {c} (n={len(sub)})",
                   edgecolor="none")
    for c in range(best_k):
        ax.scatter(km.cluster_centers_[c, 0], km.cluster_centers_[c, 1],
                   s=180, marker="X", color=cluster_colors[c % 10],
                   edgecolor="black", linewidth=1.2, zorder=5)
    ax.set_xlabel("z0"); ax.set_ylabel("z1")
    ax.set_title(f"Kincore-None chains clustered (k={best_k}, sil={best_s:.2f})")
    ax.legend(fontsize=8, frameon=False, loc="best")

    # Right: Kincore-labelled chains coloured by dihedral, with the
    # None-cluster centroids overlaid.
    ax = axes[1]
    ax.scatter(df["z0"], df["z1"], s=2, alpha=0.1, color="#dddddd",
               edgecolor="none")
    for cls, color in DIH_COLORS.items():
        if cls in ("None", ""): continue
        m = df["dihedral"] == cls
        if m.sum():
            ax.scatter(df.loc[m, "z0"], df.loc[m, "z1"],
                       s=6, alpha=0.5, color=color,
                       label=f"{cls} ({int(m.sum())})",
                       edgecolor="none")
    for c in range(best_k):
        ax.scatter(km.cluster_centers_[c, 0], km.cluster_centers_[c, 1],
                   s=300, marker="X", color=cluster_colors[c % 10],
                   edgecolor="black", linewidth=1.5, zorder=5)
        ax.text(km.cluster_centers_[c, 0], km.cluster_centers_[c, 1],
                f" None-c{c}", fontsize=10, fontweight="bold",
                color=cluster_colors[c % 10])
    ax.set_xlabel("z0"); ax.set_ylabel("z1")
    ax.set_title("None-cluster centroids vs Kincore dihedral classes")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="best")
    fig.suptitle("(2) Novel conformational regions among Kincore-unlabelled chains",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "novel_regions_latent.png")
    plt.close(fig)
    print(f"  wrote novel_regions_latent.png")


# -------------------- (3) Nearest-neighbour validation ----------------------


def analysis_nn_validation(df, coords, out_dir, n_sample=300, k=5):
    print("\n=== (3) Nearest-neighbour structural validation ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    Z = df[["z0", "z1"]].to_numpy()
    rng = np.random.default_rng(25)
    sample_idx = rng.choice(len(df), size=min(n_sample, len(df)),
                             replace=False)

    from scipy.spatial import cKDTree
    tree = cKDTree(Z)

    rows = []
    for i in sample_idx:
        # k nearest in latent (excluding self)
        d, idx = tree.query(Z[i], k=k+1)
        idx = idx[1:]  # drop self
        # latent-neighbour structural RMSDs
        nn_rmsds = [per_chain_rmsd_pairs(coords[i], coords[j]) for j in idx]
        # k random non-self chains
        rand_idx = rng.choice(len(df), size=k, replace=False)
        rand_idx = rand_idx[rand_idx != i][:k]
        rand_rmsds = [per_chain_rmsd_pairs(coords[i], coords[j]) for j in rand_idx]
        rows.append({
            "chain_idx": int(i),
            "chain_key": df.iloc[i]["chain_key"],
            "median_rmsd_latent_neighbours": float(np.median(nn_rmsds)),
            "median_rmsd_random_chains": float(np.median(rand_rmsds)),
            "ratio_nn_over_random": float(np.median(nn_rmsds) / max(np.median(rand_rmsds), 1e-6)),
        })
    nn_df = pd.DataFrame(rows)
    nn_df.to_csv(out_dir / "nn_validation.csv", index=False)
    print(f"  median latent-neighbour RMSD: {nn_df['median_rmsd_latent_neighbours'].median():.2f} Å")
    print(f"  median random-chain RMSD    : {nn_df['median_rmsd_random_chains'].median():.2f} Å")
    print(f"  median ratio (lower = better): {nn_df['ratio_nn_over_random'].median():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Distributions
    ax = axes[0]
    ax.hist(nn_df["median_rmsd_latent_neighbours"], bins=40, alpha=0.65,
            color="#315f8e", label="latent-NN", edgecolor="white")
    ax.hist(nn_df["median_rmsd_random_chains"], bins=40, alpha=0.55,
            color="#c0504d", label="random", edgecolor="white")
    ax.set_xlabel("Per-chain median Cα RMSD to k=5 neighbours (Å)")
    ax.set_ylabel(f"count (n={len(nn_df)} sampled chains)")
    ax.set_title("Latent neighbourhoods are structurally closer than random")
    ax.axvline(nn_df["median_rmsd_latent_neighbours"].median(),
               color="#315f8e", ls="--", label=f"latent median {nn_df['median_rmsd_latent_neighbours'].median():.2f}")
    ax.axvline(nn_df["median_rmsd_random_chains"].median(),
               color="#c0504d", ls="--", label=f"random median {nn_df['median_rmsd_random_chains'].median():.2f}")
    ax.legend(frameon=False)

    # Per-chain scatter
    ax = axes[1]
    ax.scatter(nn_df["median_rmsd_random_chains"],
               nn_df["median_rmsd_latent_neighbours"],
               s=15, alpha=0.6, color="#315f8e", edgecolor="none")
    mn = 0; mx = max(nn_df["median_rmsd_random_chains"].max(),
                      nn_df["median_rmsd_latent_neighbours"].max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=1)
    ax.set_xlabel("median RMSD to 5 random chains (Å)")
    ax.set_ylabel("median RMSD to 5 latent-NN (Å)")
    ax.set_title("Each point = one sampled chain")
    fig.suptitle("(3) Nearest-neighbour structural validation",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "nn_validation.png")
    plt.close(fig)
    print(f"  wrote nn_validation.png")


# -------------------- (4) ABL1 escape-route case study ---------------------


ABL1_RESISTANCE = [
    "T315I", "T315A", "T315M",            # canonical gatekeeper
    "V299L", "V299C", "V299D",            # P-loop adjacent
    "F317L", "F317V", "F317I", "F317C",   # P-loop
    "F359V", "F359C", "F359I",            # A-loop hinge / activation lip
    "Y253H", "Y253F",                     # P-loop
    "E255K", "E255V",                     # P-loop
    "G250E", "G250R",                     # P-loop
    "M388L",
    "H396P", "H396R",
    "E459K", "E459V", "E459Q",
    # Additional mutations observed in v9 SEQADV (engineered + naturally-
    # occurring variants beyond the canonical TKI-resistance set):
    "D382N",   # catalytic-loop variant
    "H415P",   # near hinge
    "L445P",   # commonly engineered
    "Y393F",   # autophos site / A-loop
]


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def seqadv_to_one_letter(s: str) -> list[str]:
    """Convert 'THR->ILE@315; LYS->ARG@29' → ['T315I', 'K29R']."""
    if not s:
        return []
    out = []
    for tok in re.findall(r"([A-Z]{3})->([A-Z]{3})@(\d+)", str(s).upper()):
        a3, b3, pos = tok
        a1 = THREE_TO_ONE.get(a3); b1 = THREE_TO_ONE.get(b3)
        if a1 and b1 and a1 != b1:
            out.append(f"{a1}{pos}{b1}")
    return out


def analysis_abl1_escape(df, muts_df, out_dir):
    print("\n=== (5) ABL1 escape routes ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    abl1 = df[df["gene"] == "ABL1"].copy()
    muts_df = muts_df[muts_df["gene"] == "ABL1"].copy()
    muts_df["chain_key"] = muts_df["chain_key"].astype(str).str.upper()
    abl1["chain_key"] = abl1["chain_key"].astype(str).str.upper()

    # Merge
    abl1 = abl1.merge(muts_df[["chain_key", "title_mutation_hits",
                                "seqadv_mutations",
                                "remark_999_mutation_lines",
                                "has_any_mutation_annotation"]],
                       on="chain_key", how="left")
    abl1["mutation_token"] = ""
    # Normalise: build a one-letter token list per chain from SEQADV +
    # title hits, then match against ABL1_RESISTANCE.
    for idx in abl1.index:
        title = str(abl1.at[idx, "title_mutation_hits"] or "")
        seqadv = str(abl1.at[idx, "seqadv_mutations"] or "")
        tokens = set()
        # 1-letter tokens from title
        for tok in re.findall(r"[A-Z]\d{2,5}[A-Z]", title.upper()):
            tokens.add(tok)
        # 3-letter SEQADV → 1-letter
        for tok in seqadv_to_one_letter(seqadv):
            tokens.add(tok)
        # Pick first matching ABL1_RESISTANCE token (canonical order)
        for mut in ABL1_RESISTANCE:
            if mut in tokens:
                abl1.at[idx, "mutation_token"] = mut
                break
    abl1["category"] = abl1.apply(
        lambda r: "WT" if not r["has_any_mutation_annotation"]
        else (r["mutation_token"] if r["mutation_token"] else "other_mut"),
        axis=1)

    # Stats per mutation
    cats = abl1["category"].value_counts()
    print("ABL1 chain breakdown:")
    print(cats.to_string())

    # Δlatent per resistance mutation vs WT centroid
    wt = abl1[abl1["category"] == "WT"][["z0", "z1"]].to_numpy()
    if len(wt) < 3:
        print("  too few WT ABL1 chains; aborting")
        return
    mu_wt = wt.mean(0)
    cov_wt = np.cov(wt.T)

    rows = []
    for mut in cats.index:
        if mut == "WT" or mut == "other_mut": continue
        sub = abl1[abl1["category"] == mut]
        if len(sub) < 1: continue
        pts = sub[["z0", "z1"]].to_numpy()
        mu = pts.mean(0)
        delta = float(np.linalg.norm(mu - mu_wt))
        diff = mu - mu_wt
        try:
            mahal_sq = float(diff @ np.linalg.pinv(cov_wt) @ diff)
        except Exception:
            mahal_sq = float("nan")
        # direction of shift (z0 / z1 axis)
        rows.append({
            "mutation": mut,
            "n_chains": int(len(sub)),
            "delta_z0": float(diff[0]),
            "delta_z1": float(diff[1]),
            "delta_latent": delta,
            "mahalanobis_sigma": float(np.sqrt(max(mahal_sq, 0))),
        })
    out_df = pd.DataFrame(rows).sort_values("delta_latent", ascending=False)
    out_df.to_csv(out_dir / "abl1_escape_routes.csv", index=False)
    print(out_df.to_string(index=False))

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    # Left: all chains grey, ABL1 chains coloured by mutation
    ax = axes[0]
    ax.scatter(df["z0"], df["z1"], s=2, alpha=0.08, color="#dddddd",
               edgecolor="none")
    palette = plt.cm.tab20.colors
    # WT in blue, mutations colour-cycled
    ax.scatter(abl1.loc[abl1["category"]=="WT", "z0"],
               abl1.loc[abl1["category"]=="WT", "z1"],
               s=25, alpha=0.5, color="#4c78a8",
               label=f"WT (n={(abl1['category']=='WT').sum()})",
               edgecolor="none")
    mut_categories = [c for c in cats.index
                       if c not in ("WT", "other_mut") and cats[c] >= 1]
    for k, mut in enumerate(mut_categories[:15]):
        sub = abl1[abl1["category"] == mut]
        ax.scatter(sub["z0"], sub["z1"], s=90, alpha=0.85,
                   color=palette[k % 20], edgecolor="black", linewidth=0.8,
                   label=f"{mut} (n={len(sub)})", marker="X")
    ax.set_xlabel("z0"); ax.set_ylabel("z1")
    ax.set_title(f"ABL1 chains in v9 latent — coloured by resistance mutation\n"
                 f"WT n={(abl1['category']=='WT').sum()}, "
                 f"mutations annotated = {len(mut_categories)}")
    ax.legend(fontsize=7, frameon=False, loc="best", ncol=2)

    # Right: WT-centroid-relative escape vectors
    ax = axes[1]
    ax.scatter(0, 0, s=300, marker="X", color="#4c78a8",
               edgecolor="black", linewidth=1.5, label="WT centroid", zorder=5)
    # Plot relative positions
    for k, mut in enumerate(mut_categories[:15]):
        sub = abl1[abl1["category"] == mut]
        rel = sub[["z0", "z1"]].to_numpy() - mu_wt
        rel_mu = rel.mean(0)
        ax.scatter(rel[:, 0], rel[:, 1], s=30, alpha=0.5,
                   color=palette[k % 20], edgecolor="none")
        # Draw arrow from WT to mutation centroid
        ax.annotate("", xy=(rel_mu[0], rel_mu[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=palette[k % 20],
                                    lw=1.6, alpha=0.85))
        ax.text(rel_mu[0]*1.05, rel_mu[1]*1.05, mut, fontsize=9,
                color=palette[k % 20], fontweight="bold")
    ax.axhline(0, color="#999", lw=0.5)
    ax.axvline(0, color="#999", lw=0.5)
    ax.set_xlabel("Δz0 from WT centroid")
    ax.set_ylabel("Δz1 from WT centroid")
    ax.set_title("Escape-route vectors per resistance mutation")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("(5) ABL1 resistance mutations in v9 latent",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "abl1_escape_routes.png")
    plt.close(fig)
    print(f"  wrote abl1_escape_routes.png")


# -------------------- main ---------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-csv", required=True, type=Path)
    ap.add_argument("--combined-pdb", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest-csv", required=True, type=Path)
    ap.add_argument("--mutations-csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.latent_csv, keep_default_na=False)
    df["z0"] = pd.to_numeric(df["z0"], errors="coerce")
    df["z1"] = pd.to_numeric(df["z1"], errors="coerce")
    df = df[df["z0"].notna() & df["z1"].notna()].copy()
    df["gene"] = df["gene"].astype(str)

    coords = load_combined_pdb(args.combined_pdb)
    print(f"Loaded {len(df)} chains, coords shape {coords.shape}")
    if len(df) != coords.shape[0]:
        raise SystemExit(f"latent rows {len(df)} != PDB models {coords.shape[0]}")

    muts = pd.read_csv(args.mutations_csv, keep_default_na=False)

    analysis_within_kinase(df, args.out)
    analysis_novel_regions(df, args.out / "novel_regions",
                            args.combined_pdb, args.checkpoint)
    analysis_nn_validation(df, coords, args.out)
    analysis_abl1_escape(df, muts, args.out)

    print(f"\nAll extended analyses written to {args.out}/")


if __name__ == "__main__":
    main()
