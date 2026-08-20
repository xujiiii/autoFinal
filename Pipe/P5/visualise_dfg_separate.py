from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def norm_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace("_", "", regex=False)


def read_ca_map(pdb_path: Path, chain: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if not pdb_path.exists():
        return out
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM") or line[21] != chain:
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[16].strip() not in {"", "A"}:
                continue
            try:
                resi = int(line[22:26])
            except ValueError:
                continue
            out[resi] = np.array(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ],
                dtype=np.float32,
            )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Plot feature distance distributions for DFGin vs DFGout."
    )
    ap.add_argument(
        "--shap-top-csv",
        required=True,
        type=Path,
        help="lgbm_shap_top.csv",
    )
    ap.add_argument(
        "--conserved-csv", required=True, type=Path, help="conserved map"
    )
    ap.add_argument(
        "--manifest-csv",
        required=True,
        type=Path,
        help="manifest from pipeline 1",
    )
    ap.add_argument(
        "--full-pdb-dir", required=True, type=Path, help="PDB folder"
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--dfg-col",
        type=str,
        default="dfg_spatial",
        help="column of manifest that store dfg status",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out / "figures"
    fig_dir.mkdir(exist_ok=True)

    shap_df = pd.read_csv(args.shap_top_csv)
    top_pairs = (
        shap_df.groupby("target")
        .apply(lambda x: x.sort_values("rank").head(9))
        .reset_index(drop=True)
    )
    unique_pairs = (
        top_pairs[["resi_i", "resi_j"]]
        .drop_duplicates()
        .to_records(index=False)
    )
    unique_pairs = [(int(i), int(j)) for i, j in unique_pairs]
    pair_to_idx = {pair: idx for idx, pair in enumerate(unique_pairs)}

    print(f"{len(unique_pairs)} Top SHAP distance pair")

    conserved = pd.read_csv(args.conserved_csv, keep_default_na=False)
    conserved["chain_key"] = norm_key(conserved["chain_key"])

    manifest = pd.read_csv(args.manifest_csv, keep_default_na=False)
    manifest["chain_key"] = manifest["chain_key"].astype(str).str.upper()

    if args.dfg_col not in manifest.columns:
        raise KeyError(
            f"{args.dfg_col} not in {args.manifest_csv} Columns:"
            f" {list(manifest.columns)}"
        )

    manifest = manifest.dropna(subset=[args.dfg_col]).copy()
    valid_states = {"DFGin", "DFGout", "DFG-in", "DFG-out"}
    manifest = manifest[
        manifest[args.dfg_col].astype(str).isin(valid_states)
    ].reset_index(drop=True)

    print(f"All dataa can be used: {len(manifest)}")

    chain_maps = (
        conserved.groupby("chain_key").apply(
            lambda g: dict(
                zip(g["braf_resi"].astype(int), g["pdb_resi"].astype(int))
            )
        )
    ).to_dict()


    n_samples = len(manifest)
    X_dist = np.full((n_samples, len(unique_pairs)), np.nan, dtype=np.float32)

    for ii in range(n_samples):
        if ii % 200 == 0:
            print(f"Computing distance: {ii}/{n_samples}")
        row = manifest.iloc[ii]
        key = row["chain_key"]
        cmap = chain_maps.get(key)
        if cmap is None:
            continue

        cas = read_ca_map(
            args.full_pdb_dir / f"{row['pdb']}.pdb", row["chain"]
        )
        for jj, (ri, rj) in enumerate(unique_pairs):
            pi, pj = cmap.get(ri), cmap.get(rj)
            if pi is None or pj is None:
                continue
            ci, cj = cas.get(pi), cas.get(pj)
            if ci is None or cj is None:
                continue
            X_dist[ii, jj] = float(np.linalg.norm(ci - cj))

    dfg_labels = manifest[args.dfg_col].astype(str).values

    color_map = {
        "DFGin": "#315f8e", 
        "DFGout": "#c0504d", 
    }

    target_classes = ["DFGin", "DFGout"]

    for tg in ("z0", "z1"):
        sub = top_pairs[top_pairs["target"] == tg].sort_values("rank").head(9)
        if len(sub) == 0:
            continue

        raw_export_rows = []
        hist_density_rows = []

        for k, (_, row_data) in enumerate(sub.iterrows()):
            ri, rj = int(row_data["resi_i"]), int(row_data["resi_j"])
            fi = pair_to_idx[(ri, rj)]
            rank = int(row_data["rank"])
            feature_name = f"d_{ri}_{rj}"

            fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)

            for cls in target_classes:
                cls_mask = (dfg_labels == cls) | (
                    dfg_labels == cls.replace("DFG", "DFG-")
                )
                color = color_map[cls]

                m = cls_mask & (~np.isnan(X_dist[:, fi]))
                if m.sum() == 0:
                    continue

                distances = X_dist[m, fi]

                for val in distances:
                    raw_export_rows.append({
                        "target": tg,
                        "rank": rank,
                        "feature": feature_name,
                        "resi_i": ri,
                        "resi_j": rj,
                        "dfg_class": cls,
                        "distance_angstrom": float(val),
                    })

                densities, bin_edges, _ = ax.hist(
                    distances,
                    bins=30,
                    alpha=0.55,
                    density=True,
                    label=f"{cls} (n={m.sum()})",
                    color=color,
                )

                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

                for bc, den in zip(bin_centers, densities):
                    hist_density_rows.append({
                        "target": tg,
                        "rank": rank,
                        "feature": feature_name,
                        "resi_i": ri,
                        "resi_j": rj,
                        "dfg_class": cls,
                        "bin_center_d": float(bc),
                        "density": float(den),
                    })


            ax.set_xlabel(f"d({ri}, {rj}) [Å]", fontsize=16)
            ax.set_ylabel("Density", fontsize=16)
            ax.set_title(
                f"Rank #{rank}: d({ri}, {rj})", fontsize=16, fontweight="bold"
            )
            ax.legend(fontsize=12, frameon=False)

            fig.tight_layout()

            single_fig_name = f"top_feature_rank{rank}_{tg}_d{ri}_{rj}"
            fig.savefig(fig_dir / f"{single_fig_name}.png")
            fig.savefig(fig_dir / f"{single_fig_name}.pdf")
            plt.close(fig)


        pd.DataFrame(raw_export_rows).to_csv(
            args.out / f"top_feature_raw_distances_{tg}.csv", index=False
        )
        pd.DataFrame(hist_density_rows).to_csv(
            args.out / f"top_feature_hist_density_{tg}.csv", index=False
        )
        print(f"Save {tg} (DFGin vs DFGout) plot and CSV records")

    print(f"Done, all outputs under: {args.out}")


if __name__ == "__main__":
    main()