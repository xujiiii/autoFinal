"""Evaluate every v9 variant / seed / gene-split run uniformly.

For each run dir produced by ``train_v9_variant.py`` (best.ckpt,
final.pkl, train_idx.txt, test_idx.txt) computes:

  - Reconstruction per-chain Cα RMSD on the run's own TEST split
  - Mean, median, p90 RMSD
  - Variance explained on test
  - Reconstruction RMSD by Kincore dihedral / DFG-spatial
  - Linear-probe accuracy: train a logistic regression on the latent
    to predict DFG-spatial and Kincore dihedral

Outputs:
  <out>/variant_summary.csv          (one row per run)
  <out>/variant_per_chain.csv        (joined per-chain RMSD across runs)
  <out>/probes.csv                   (linear-probe metrics per run)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_mutual_info_score, balanced_accuracy_score)
from sklearn.preprocessing import LabelEncoder


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


# def reconstruct(coords_raw: np.ndarray, ckpt_path: Path,
#                 preprocess: str, params: dict):
#     """Returns (recon_raw, latent) of shape (N, n_atoms, 3) and (N, 2).

#     preprocess: 'center' or 'standardise'
#     params: dict from final.pkl with preprocess_mean / preprocess_std if needed
#     """
#     n, n_atoms, _ = coords_raw.shape
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     from molearn.models.small_foldingnet import Small_AutoEncoder
#     net = Small_AutoEncoder(out_points=n_atoms).to(device)
#     state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
#     sd = state.get("model_state_dict", state.get("state_dict", state))
#     net.load_state_dict(sd)
#     net.eval()

#     if preprocess == "center":
#         centroids = coords_raw.mean(axis=1, keepdims=True)
#         coords_pp = coords_raw - centroids
#         def unpp(y):
#             return y + centroids
#     else:
#         mean = float(params["preprocess_mean"])
#         std = float(params["preprocess_std"])
#         coords_pp = (coords_raw - mean) / std
#         def unpp(y):
#             return y * std + mean

#     X = torch.from_numpy(coords_pp).float().to(device)
#     Y, Z = [], []
#     with torch.no_grad():
#         for i in range(0, n, 128):
#             batch = X[i:i + 128]
            
#             z = net.encode(batch)
#             Z.append(z.cpu().numpy().reshape(z.shape[0], -1))
#             out = net(batch)[:, :n_atoms, :]
#             Y.append(out.cpu().numpy())
#     Y = np.concatenate(Y, axis=0)
#     Z = np.concatenate(Z, axis=0)
#     return unpp(Y), Z


def reconstruct(coords_raw: np.ndarray, ckpt_path: Path,
                preprocess: str, params: dict):
    """Returns (recon_raw, latent) of shape (N, n_atoms, 3) and (N, 2).

    preprocess: 'center' or 'standardise'
    params: dict from final.pkl with preprocess_mean / preprocess_std if needed
    """
    n, n_atoms, _ = coords_raw.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from molearn.models.small_foldingnet import Small_AutoEncoder
    net = Small_AutoEncoder(out_points=n_atoms).to(device)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state.get("state_dict", state))
    net.load_state_dict(sd)
    net.eval()

    if preprocess == "center":
        centroids = coords_raw.mean(axis=1, keepdims=True)
        coords_pp = coords_raw - centroids
        def unpp(y):
            return y + centroids
    else:
        mean = float(params["preprocess_mean"])
        std = float(params["preprocess_std"])
        coords_pp = (coords_raw - mean) / std
        def unpp(y):
            return y * std + mean

    X = torch.from_numpy(coords_pp).float().to(device)
    Y, Z = [], []
    with torch.no_grad():
        for i in range(0, n, 128):
            # Shape of batch_raw: (B, n_atoms, 3)
            batch_raw = X[i:i + 128]
            
            # Permute to (B, 3, n_atoms) for FoldingNet / Small_AutoEncoder
            batch = batch_raw.permute(0, 2, 1)

            # Encoding & Decoding
            z = net.encode(batch)
            out = net(batch)  # Output shape: (B, 3, n_atoms)

            # Permute output back to (B, n_atoms, 3)
            out = out.permute(0, 2, 1)[:, :n_atoms, :]

            Z.append(z.cpu().numpy().reshape(z.shape[0], -1))
            Y.append(out.cpu().numpy())

    Y = np.concatenate(Y, axis=0)
    Z = np.concatenate(Z, axis=0)
    return unpp(Y), Z

def per_chain_rmsd(a, b):
    """Kabsch-superposed per-chain Cα RMSD (a = recon, b = target),
    shapes (N_chains, n_atoms, 3).

    FIX 2026-06-20: previously this returned the RAW (unaligned)
    coordinate RMSD. An autoencoder can reconstruct a loop's shape
    correctly but place it at a slightly wrong absolute position /
    orientation (especially for held-out families it never saw), which
    raw RMSD charges as error — inflating every number and grossly
    overstating the gene-group-holdout error (raw 15.7 Å vs aligned
    9.7 Å). Superpose each chain before measuring shape error.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.empty(a.shape[0], dtype=np.float64)
    for i in range(a.shape[0]):
        P = a[i] - a[i].mean(0)
        Q = b[i] - b[i].mean(0)
        V, S, Wt = np.linalg.svd(P.T @ Q)
        d = np.sign(np.linalg.det(V @ Wt))
        P = P @ (V @ np.diag([1.0, 1.0, d]) @ Wt)
        out[i] = np.sqrt(((P - Q) ** 2).sum(axis=-1).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-pdb", required=True, type=Path)
    ap.add_argument("--manifest-csv", required=True, type=Path)
    ap.add_argument("--run-dirs", required=True, nargs="+", type=Path,
                    help="Directories produced by train_v9_variant.py")
    ap.add_argument("--labels", required=True, nargs="+",
                    help="Short label for each run dir (same order).")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if len(args.run_dirs) != len(args.labels):
        raise SystemExit("--run-dirs and --labels must have same length")

    args.out.mkdir(parents=True, exist_ok=True)
    coords = load_combined_pdb(args.combined_pdb)
    n, n_atoms, _ = coords.shape
    manifest = pd.read_csv(args.manifest_csv, keep_default_na=False)
    if len(manifest) != n:
        raise SystemExit(f"PDB has {n} models, manifest {len(manifest)} rows")

    summary_rows = []
    probe_rows = []
    per_chain_cols = {}

    for run, label in zip(args.run_dirs, args.labels):
        ckpt = run / "best.ckpt"
        meta = run / "final.pkl"
        train_idx = np.loadtxt(run / "train_idx.txt", dtype=int)
        test_idx = np.loadtxt(run / "test_idx.txt", dtype=int)
        with meta.open("rb") as f:
            params = pickle.load(f)
        preprocess = params["preprocess"]
        print(f"\n=== {label}  ({preprocess}, batch={params['batch_size']}, seed={params['seed']}) ===")

        recon, latent = reconstruct(coords, ckpt, preprocess, params)
        rmsd = per_chain_rmsd(coords, recon)

        # Variance explained on the run's own test split.
        X_test = coords[test_idx]
        total_var = ((X_test - X_test.mean(axis=0)) ** 2).sum() / len(test_idx)
        recon_var = (rmsd[test_idx] ** 2 * n_atoms).mean()
        ev = 1 - recon_var / total_var

        summary_rows.append({
            "run": label,
            "preprocess": preprocess,
            "batch_size": params["batch_size"],
            "seed": params["seed"],
            "gene_group_split": params.get("gene_group_split", False),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "variance_explained_test": float(ev) * 100,
            "rmsd_mean": float(rmsd[test_idx].mean()),
            "rmsd_median": float(np.median(rmsd[test_idx])),
            "rmsd_p90": float(np.quantile(rmsd[test_idx], 0.90)),
            "rmsd_max": float(rmsd[test_idx].max()),
        })
        per_chain_cols[f"rmsd_{label}_A"] = rmsd

        # Linear probes (train on train_idx, eval on test_idx).
        le_dfg = LabelEncoder()
        le_dih = LabelEncoder()
        m_dfg = manifest["dfg_spatial"].astype(str).fillna("None")
        m_dih = manifest["dihedral"].astype(str).fillna("None")
        ok = (m_dfg != "") & (m_dfg != "None")
        ok2 = (m_dih != "") & (m_dih != "None")
        # DFG probe
        try:
            y_dfg = le_dfg.fit_transform(m_dfg[ok])
            tr = np.intersect1d(np.where(ok)[0], train_idx)
            te = np.intersect1d(np.where(ok)[0], test_idx)
            clf = LogisticRegression(max_iter=1000, n_jobs=-1)
            clf.fit(latent[tr], le_dfg.transform(m_dfg.iloc[tr]))
            yhat = clf.predict(latent[te])
            ytrue = le_dfg.transform(m_dfg.iloc[te])
            dfg_bal = balanced_accuracy_score(ytrue, yhat)
            dfg_raw = (yhat == ytrue).mean()
        except Exception as e:
            dfg_bal = dfg_raw = float("nan")
            print(f"  DFG probe failed: {e}")
        # dihedral probe
        try:
            y_dih = le_dih.fit_transform(m_dih[ok2])
            tr2 = np.intersect1d(np.where(ok2)[0], train_idx)
            te2 = np.intersect1d(np.where(ok2)[0], test_idx)
            clf2 = LogisticRegression(max_iter=1000, n_jobs=-1)
            clf2.fit(latent[tr2], le_dih.transform(m_dih.iloc[tr2]))
            yhat2 = clf2.predict(latent[te2])
            ytrue2 = le_dih.transform(m_dih.iloc[te2])
            dih_bal = balanced_accuracy_score(ytrue2, yhat2)
            dih_raw = (yhat2 == ytrue2).mean()
            dih_ami = adjusted_mutual_info_score(ytrue2, yhat2)
        except Exception as e:
            dih_bal = dih_raw = dih_ami = float("nan")
            print(f"  dihedral probe failed: {e}")

        probe_rows.append({
            "run": label,
            "DFG_balanced_acc": float(dfg_bal),
            "DFG_raw_acc": float(dfg_raw),
            "dihedral_balanced_acc": float(dih_bal),
            "dihedral_raw_acc": float(dih_raw),
            "dihedral_AMI": float(dih_ami),
        })
        print(f"  EV={ev*100:.1f}%  median RMSD={np.median(rmsd[test_idx]):.2f} Å  "
              f"DFG bal={dfg_bal:.3f}  dihedral bal={dih_bal:.3f}  AMI={dih_ami:.3f}")

    pd.DataFrame(summary_rows).to_csv(args.out / "variant_summary.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(args.out / "probes.csv", index=False)
    per = manifest.copy()
    for col, vec in per_chain_cols.items():
        per[col] = vec
    per.to_csv(args.out / "variant_per_chain.csv", index=False)

    print("\n========== SUMMARY ==========")
    print(pd.DataFrame(summary_rows)[
        ["run", "preprocess", "batch_size", "seed", "gene_group_split",
         "variance_explained_test", "rmsd_median", "rmsd_p90"]
    ].to_string(index=False))
    print("\n========== PROBES ==========")
    print(pd.DataFrame(probe_rows).to_string(index=False))


if __name__ == "__main__":
    main()
