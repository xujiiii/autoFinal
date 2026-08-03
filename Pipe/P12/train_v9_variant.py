"""Variant trainer for v9 ablations.

Configurable preprocessing (per-chain center vs global standardise) and
batch size. Otherwise identical to ``train_v6_foldingnet_ca.py`` (our
custom CA-only training loop, Adam lr=1e-3, MSE loss in the chosen
preprocessing space).

Used to isolate which factor explains the variance-explained gap
between our baseline AE (94.7 %) and the mey wrapper (~85 %):

  --preprocess center     batch 16  →  variant: baseline (= existing)
  --preprocess standardise batch 16 →  variant A (preprocessing only)
  --preprocess center     batch 8   →  variant B (batch only)
  --preprocess standardise batch 8  →  variant C (both)

Also useful for seed-robustness: --seed 101 / 202 / 303 with otherwise
identical setup to baseline.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-pdb", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--preprocess", choices=("center", "standardise"),
                    default="center")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="L2 weight decay. molearn Trainer uses 1e-4.")
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--valid-ratio", type=float, default=0.1)
    ap.add_argument("--gene-group-split", action="store_true",
                    help="Use gene-level held-out split (no gene appears "
                         "in both train and test) instead of random split.")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "xbb_foldingnet_checkpoints").mkdir(exist_ok=True)

    from molearn.models.small_foldingnet import Small_AutoEncoder

    coords = load_combined_pdb(args.combined_pdb)
    n, n_atoms, _ = coords.shape
    print(f"Loaded {n} chains × {n_atoms} CAs.")

    # Preprocessing.
    if args.preprocess == "center":
        centroids = coords.mean(axis=1, keepdims=True)
        coords_pp = coords - centroids
        inv_pp = lambda y, i: y + centroids[i]  # for diagnostic only
        std_for_loss = coords_pp.std()
        print(f"[preprocess=center] coord std: {std_for_loss:.2f} Å")
    else:
        mean = float(coords.mean())
        std = float(coords.std())
        coords_pp = (coords - mean) / std
        inv_pp = lambda y, i: y * std + mean
        print(f"[preprocess=standardise] global mean={mean:.2f}, "
              f"std={std:.2f}")

    # Split.
    rng = np.random.default_rng(args.seed)
    if args.gene_group_split:
        mani = pd.read_csv(args.manifest, keep_default_na=False)
        genes = mani["gene"].fillna("").astype(str).values
        unique_genes = sorted(set(genes))
        rng.shuffle(np.asarray(unique_genes))
        n_test_genes = max(1, int(len(unique_genes) * args.valid_ratio))
        test_genes = set(unique_genes[:n_test_genes])
        test_mask = np.array([g in test_genes for g in genes])
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        print(f"[gene-group split] {len(test_genes)} held-out genes → "
              f"{len(test_idx)} test, {len(train_idx)} train chains")
    else:
        perm = rng.permutation(n)
        n_test = int(n * args.valid_ratio)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        print(f"[random split, seed={args.seed}] "
              f"train {len(train_idx)}, test {len(test_idx)}")

    np.savetxt(out / "train_idx.txt", train_idx, fmt="%d")
    np.savetxt(out / "test_idx.txt", test_idx, fmt="%d")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ------------------------------------------------------------------
    # 【修改点 1】：转置维度从 (N, n_atoms, 3) 变为 (N, 3, n_atoms)
    # ------------------------------------------------------------------
    X = torch.from_numpy(coords_pp).float().transpose(1, 2).to(device)
    X_train = X[train_idx]; X_valid = X[test_idx]

    torch.manual_seed(args.seed)
    net = Small_AutoEncoder(out_points=n_atoms).to(device)
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(net.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)

    train_loader = DataLoader(
        TensorDataset(X_train), batch_size=args.batch_size,
        shuffle=True, drop_last=False,
    )

    log_path = out / "xbb_foldingnet_checkpoints" / "log_file.dat"
    log_path.write_text("epoch\ttrain_mse\tvalid_mse\n")
    best_valid = float("inf")
    for ep in range(args.epochs):
        net.train()
        train_loss = 0.0
        n_batches = 0
        for (xb,) in train_loader:
            opt.zero_grad()
            # --------------------------------------------------------------
            # 【修改点 2】：在 dim 2 (点数维度) 上切片截断：[:, :, :n_atoms]
            # --------------------------------------------------------------
            yb = net(xb)[:, :, :n_atoms]
            loss = ((yb - xb) ** 2).mean()
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            n_batches += 1
        train_loss /= max(1, n_batches)
        net.eval()
        with torch.no_grad():
            yv = []
            for i in range(0, X_valid.shape[0], 128):
                # ----------------------------------------------------------
                # 【修改点 3】：验证集预测同样修改切片维度为 dim 2
                # ----------------------------------------------------------
                yv.append(net(X_valid[i:i + 128])[:, :, :n_atoms])
            yv = torch.cat(yv, dim=0)
            valid_loss = float(((yv - X_valid) ** 2).mean().item())
        with log_path.open("a") as f:
            f.write(f"{ep}\t{train_loss:.6f}\t{valid_loss:.6f}\n")
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(net.state_dict(), out / "best.ckpt")

    with (out / "final.pkl").open("wb") as f:
        pickle.dump({"best_valid": best_valid, "epochs": args.epochs,
                     "preprocess": args.preprocess,
                     "batch_size": args.batch_size, "lr": args.lr,
                     "seed": args.seed,
                     "gene_group_split": args.gene_group_split,
                     "n_atoms": n_atoms,
                     "preprocess_mean":
                         float(coords.mean()) if args.preprocess == "standardise" else None,
                     "preprocess_std":
                         float(coords.std()) if args.preprocess == "standardise" else None,
                     }, f)
    print(f"Done. best_valid={best_valid:.4f}")


if __name__ == "__main__":
    main()
